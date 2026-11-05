//! DeepSeek-V4.1-Flash behind TensorFold 1.0's lane core (core/lanes): the round loop picks each round's streams and
//! windows and keeps the accepted rows; this backend runs them on the port's model (model.zig) with the served lane's
//! arithmetic. Each stream has a pool slot (its window rings, positional stores and drafter rings) and an extent of
//! the shared compressed plane (first fit, chunk-aligned, as large as its prompt, its budget and a round's rows), and
//! its ids by position on both ranks (Engram's n-grams). verify is one round over every window (round.zig) with each
//! row drawn by the served sampler (sampling.zig) at its keyed position, and the drafting streams' window rows
//! absorbed into their drafter rings at once (the served eager absorb: the rows past the kept ones sit where the
//! drafter never reads, and the next window writes over them); draft is one batched drafter pass (draft.zig) for
//! every stream asked, the drafts held for the next round with their confidences, whose sigmoids are each draft's
//! chance of landing once the earlier ones did (probabilities: their running products, the chance that a draft and
//! every earlier one land; the round loop's allocation trims the windows by them). The caches are
//! positional, so keep only checks that a path is a prefix. Rank 0 runs the core; before each primitive it sends rank
//! 1 the primitive's inputs over the ranks' link, and rank 1 (follow) runs the same primitive: both ranks issue the
//! same kernels and collectives in the same order. The drafted reply is the serial one: every row is drawn with its
//! stream's key at its own position, whatever the round's composition.
const std = @import("std");
const lanes = @import("lanes");
const model = @import("model.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const sampling = @import("sampling.zig");
const link = @import("link.zig");
const coll = @import("coll.zig");
const gpu = @import("gpu");
/// The vision tower's type: CUDA builds only (gpu.features.vision); an empty type a HIP build never fills.
pub const VisionTower = if (gpu.features.vision) @import("vision.zig").Vision else opaque {};

const be = lanes.backend;
const grammar = lanes.grammar;
const Model = model.Model;
const max_streams = model.max_streams;
const max_rows = round.max_rows;

/// A window of the pending row and the drafter's block: the widest one the gates checked row for row.
pub const window_rows = 6;

/// The image rows a fill_images frame carries at most.
pub const image_piece = 32 << 10;

/// A gate's diagnostic (the tools' --logits-sha 1): each prompt's logits' sha256 and rows on stderr.
pub var logits_sha = false;
/// The most prompt plus max_tokens tokens a request may hold in a pool of `pool` positions: a stream's extent is the
/// request's tokens, a round's rows and two more, rounded up to whole prompt chunks, and must fit the pool.
pub fn usableContext(pool: usize, round_rows: usize) usize {
    return (pool / model.chunk_rows * model.chunk_rows) -| (round_rows + 2);
}

test "the usable context is the largest request whose extent fits the pool" {
    try std.testing.expectEqual(@as(usize, 8174), usableContext(8192, 16));
    try std.testing.expectEqual(@as(usize, 8174), usableContext(10000, 16));
    for ([_]usize{ 8192, 10000, 16384, 65487 }) |pool| {
        const u = usableContext(pool, 16);
        try std.testing.expect(std.mem.alignForward(usize, u + 16 + 2, model.chunk_rows) <= pool);
        try std.testing.expect(std.mem.alignForward(usize, u + 1 + 16 + 2, model.chunk_rows) > pool);
    }
}

/// TF_DS_TRACE_ROUNDS (rank 0): one stderr line a verify window: the stream's cache length, its rows, their ids (the
/// pending token, then the drafts) and the tokens drawn for them; at 2 also each row's two best tokens and logits
/// (the round's logits fetched to the host: a diagnostic, slower).
pub var trace_rounds: u8 = 0;

/// A row's two best tokens and their logits (the first wins ties, as the device argmax).
fn topTwo(row: []const f32) [2]struct { u32, f32 } {
    var best: [2]struct { u32, f32 } = .{ .{ 0, -std.math.inf(f32) }, .{ 0, -std.math.inf(f32) } };
    for (row, 0..) |v, i| {
        if (v > best[0][1]) {
            best[1] = best[0];
            best[0] = .{ @intCast(i), v };
        } else if (v > best[1][1]) best[1] = .{ @intCast(i), v };
    }
    return best;
}

/// A gate's dump (the tools' --logits-dump DIR): each prompt's logits (its row len - 1) as raw little-endian f32 in
/// DIR/<name>.f32 and a JSON beside it (DIR/<name>.json: rows, columns, vocab_size, sha256, the top two ids and logits).
pub var logits_dump: ?[]const u8 = null;

/// The primitives rank 0 sends rank 1 (each frame: kind u8, step u64, its fields little-endian).
pub const Kind = enum(u8) { fill_begin = 1, fill_chunk = 2, verify = 3, pass = 4, snap_save = 5, snap_restore = 6, snap_drop = 7, fill_images = 8, done = 9 };

/// A frame being written.
pub const Wire = struct {
    buf: std.ArrayList(u8) = .empty,

    pub fn begin(w: *Wire, gpa: std.mem.Allocator, kind: Kind, step: u64) !void {
        w.buf.clearRetainingCapacity();
        try w.int(gpa, u8, @backingInt(kind));
        try w.int(gpa, u64, step);
    }

    pub fn int(w: *Wire, gpa: std.mem.Allocator, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.buf.appendSlice(gpa, &b);
    }

    pub fn ids(w: *Wire, gpa: std.mem.Allocator, v: []const i32) !void {
        try w.int(gpa, u32, @intCast(v.len));
        try w.buf.ensureUnusedCapacity(gpa, v.len * 4);
        for (v) |x| try w.int(gpa, i32, x);
    }

    pub fn deinit(w: *Wire, gpa: std.mem.Allocator) void {
        w.buf.deinit(gpa);
    }
};

/// A frame being read.
pub const Reader = struct {
    b: []const u8,
    at: usize = 0,

    pub fn int(r: *Reader, comptime T: type) !T {
        const n = @sizeOf(T);
        if (r.at + n > r.b.len) return error.ShortFrame;
        const v = std.mem.readInt(T, r.b[r.at..][0..n], .little);
        r.at += n;
        return v;
    }

    /// Every byte of the frame was read (else BadFrame: a frame with more fields than this build reads, a mixed pair).
    pub fn end(r: *const Reader) !void {
        if (r.at != r.b.len) return error.BadFrame;
    }

    /// The frame's bytes after those read.
    pub fn rest(r: *Reader) []const u8 {
        const b = r.b[r.at..];
        r.at = r.b.len;
        return b;
    }

    /// A list of ids written by Wire.ids, into `out` (replacing it).
    pub fn ids(r: *Reader, gpa: std.mem.Allocator, out: *std.ArrayList(i32)) !void {
        const n = try r.int(u32);
        try out.resize(gpa, n);
        for (out.items) |*x| x.* = try r.int(i32);
    }
};

/// A window of a round as both ranks build it: its stream's slot and extent, its first position, whether its rows'
/// taps go into the drafter, its rows' ids.
pub const Win = struct { slot: usize, base: usize, end: usize, start: usize, absorb: bool, ids: []const i32 };

/// The rows of a round (round.Rows' slices point here).
pub const Built = struct {
    ids: [max_rows]i64 = undefined,
    pos: [max_rows]i64 = undefined,
    slots: [max_rows]i64 = undefined,
    bases: [max_rows]i64 = undefined,
    ends: [max_rows]i64 = undefined,
    wins: [max_streams]round.Window = undefined,
    absorb: [max_streams]bool = undefined,
    r: usize = 0,
    n: usize = 0,

    /// Each window's ids written at its positions in its slot's sequence (the rows past them dropped: the rejected
    /// rows of the last round), then the round's rows in window order.
    pub fn build(b: *Built, gpa: std.mem.Allocator, seqs: *[max_streams]std.ArrayList(i32), wins: []const Win) !round.Rows {
        if (wins.len == 0 or wins.len > max_streams) return error.BadRound;
        b.r = 0;
        b.n = wins.len;
        for (wins, 0..) |w, i| {
            if (w.slot >= max_streams or w.ids.len == 0) return error.BadRound;
            if (b.r + w.ids.len > max_rows) return error.WindowTooWide;
            const seq = &seqs[w.slot];
            if (seq.items.len < w.start) return error.SequenceGap;
            try seq.resize(gpa, w.start + w.ids.len);
            @memcpy(seq.items[w.start..], w.ids);
            for (w.ids, 0..) |id, j| {
                b.ids[b.r + j] = id;
                b.pos[b.r + j] = @intCast(w.start + j);
                b.slots[b.r + j] = @intCast(w.slot);
                b.bases[b.r + j] = @intCast(w.base);
                b.ends[b.r + j] = @intCast(w.end);
            }
            b.wins[i] = .{ .row = b.r, .n = w.ids.len, .seq = seq.items[0 .. w.start + w.ids.len] };
            b.absorb[i] = w.absorb;
            b.r += w.ids.len;
        }
        const R = b.r;
        return .{ .ids = b.ids[0..R], .pos = b.pos[0..R], .slots = b.slots[0..R], .bases = b.bases[0..R], .ends = b.ends[0..R], .windows = b.wins[0..b.n] };
    }
};

/// The first chunk-aligned extent of `size` positions in [0, cap) that overlaps none of `used`.
/// Whether [lo, hi) meets any of `used`.
fn clashes(used: []const [2]usize, lo: usize, hi: usize) bool {
    for (used) |u| if (lo < u[1] and u[0] < hi) return true;
    return false;
}

/// Whether one of the prompt cache's marks has its chunk end at `at` (each kept at the chunk end at or before it).
fn markEnds(marks: []const u32, at: usize) bool {
    for (marks) |w| if (w / model.chunk_rows * model.chunk_rows == at) return true;
    return false;
}

/// The first extent of `size` in [0, cap) clear of `live` and of the kept states ([base, at, end] each): clear of their
/// whole extents (rows and tail) where one fits so, else of their rows only.
pub fn placeSoft(gpa: std.mem.Allocator, live: []const [2]usize, kept: []const [3]usize, size: usize, cap: usize) !?usize {
    var used: std.ArrayList([2]usize) = .empty;
    defer used.deinit(gpa);
    try used.appendSlice(gpa, live);
    for (kept) |k| try used.append(gpa, .{ k[0], @max(k[2], k[0] + k[1]) });
    if (place(used.items, size, cap)) |b| return b;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| try used.append(gpa, .{ k[0], k[0] + k[1] });
    return place(used.items, size, cap);
}

test "a kept state's tail stays free while a placement fits elsewhere" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    const kept = [_][3]usize{.{ 0, 4 * c, 5 * c }}; // rows [0, 4c), its stream's extent to 5c
    try std.testing.expectEqual(@as(?usize, 5 * c), try placeSoft(gpa, &.{}, &kept, 4 * c, 16 * c));
    // no room clear of the tail: the rows alone are held back
    const live = [_][2]usize{.{ 9 * c, 16 * c }};
    try std.testing.expectEqual(@as(?usize, 4 * c), try placeSoft(gpa, &live, &kept, 5 * c, 16 * c));
    try std.testing.expectEqual(@as(?usize, null), try placeSoft(gpa, &live, &kept, 6 * c, 16 * c));
}

/// Whether, with an extent of `size` at `b` taken besides `used`, another extent of `size` still fits in [0, cap).
pub fn roomAfter(gpa: std.mem.Allocator, used: []const [2]usize, b: usize, size: usize, cap: usize) !bool {
    const all = try gpa.alloc([2]usize, used.len + 1);
    defer gpa.free(all);
    @memcpy(all[0..used.len], used);
    all[used.len] = .{ b, b + size };
    return place(all, size, cap) != null;
}

test "a copy that fills the pool leaves no room for another request of its size" {
    const c = model.chunk_rows;
    // three live extents of 4 chunks in a 16-chunk pool: a fourth fits, and leaves no room for a fifth
    const used = [_][2]usize{ .{ 0, 4 * c }, .{ 4 * c, 8 * c }, .{ 8 * c, 12 * c } };
    const gpa = std.testing.allocator;
    const b = place(&used, 4 * c, 16 * c).?;
    try std.testing.expectEqual(@as(usize, 12 * c), b);
    try std.testing.expect(!try roomAfter(gpa, &used, b, 4 * c, 16 * c));
    // two live extents: the third leaves room for a fourth
    const b2 = place(used[0..2], 4 * c, 16 * c).?;
    try std.testing.expect(try roomAfter(gpa, used[0..2], b2, 4 * c, 16 * c));
}

pub fn place(used: []const [2]usize, size: usize, cap: usize) ?usize {
    var base: usize = 0;
    while (base + size <= cap) {
        var clash: ?usize = null;
        for (used) |u| {
            if (base < u[1] and u[0] < base + size) clash = @max(clash orelse 0, u[1]);
        }
        const past = clash orelse return base;
        base = std.mem.alignForward(usize, past, model.chunk_rows);
    }
    return null;
}

/// The token a row's logits draw at absolute position `position`: greedy without sampling (or at temperature 0),
/// else the served sampler with the stream's key.
pub fn draw(row: []const f32, position: u64, s: ?lanes.Sampling) !u32 {
    const sm = s orelse return @intCast(sampling.argmax(row));
    if (sm.temperature <= 0) return @intCast(sampling.argmax(row));
    const t = try sampling.sampleRow(row, position, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p });
    return @intCast(t);
}

/// --logits-dump: a prompt's logits row in DIR/<name>.f32 and its JSON in DIR/<name>.json (the top two among the
/// first `vocab` columns).
fn dumpLogits(io: std.Io, path: []const u8, name: []const u8, rows: usize, vocab: usize, logits: []const f32, sha: []const u8) !void {
    if (name.len == 0 or std.mem.indexOfAny(u8, name, "/\\\"") != null) return error.BadDumpName;
    const t = topTwo(logits[0..@min(vocab, logits.len)]);
    try std.Io.Dir.cwd().createDirPath(io, path);
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var nb: [256]u8 = undefined;
    var f = try dir.createFile(io, try std.fmt.bufPrint(&nb, "{s}.f32", .{name}), .{});
    defer f.close(io);
    try f.writePositionalAll(io, std.mem.sliceAsBytes(logits), 0);
    var jb: [768]u8 = undefined;
    const text = try std.fmt.bufPrint(&jb, "{{\"name\": \"{s}\", \"rows\": {d}, \"row\": {d}, \"columns\": {d}, \"vocab_size\": {d}, \"dtype\": \"f32\", \"sha256\": \"{s}\", \"top\": [[{d}, {d}], [{d}, {d}]]}}\n", .{ name, rows, rows - 1, logits.len, vocab, sha, t[0][0], t[0][1], t[1][0], t[1][1] });
    try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb, "{s}.json", .{name}), .data = text });
}

test "topTwo: the two largest logits, in order" {
    const ids = struct {
        fn of(b: anytype) [2]u32 {
            return .{ b[0][0], b[1][0] };
        }
    }.of;
    const b = topTwo(&.{ 0.5, 3.0, -1.0, 2.0, 1.0 });
    try std.testing.expectEqual([2]u32{ 1, 3 }, ids(b));
    try std.testing.expectEqual([2]f32{ 3.0, 2.0 }, [2]f32{ b[0][1], b[1][1] });
    try std.testing.expectEqual([2]u32{ 1, 3 }, ids(topTwo(&.{ 0.5, 3.0, 2.0, 3.0 }))); // a tie: the first is the top, the second the next
    try std.testing.expectEqual([2]u32{ 1, 2 }, ids(topTwo(&.{ 1.0, 4.0, 2.0 })));
}

/// `draw` among the tokens the stream's grammar allows here (a free draw where it constrains nothing): greedy takes
/// the best allowed token, sampling the served sampler over the best top_k allowed ones.
pub fn drawAllowed(c: *grammar.Constraint, row: []const f32, position: u64, s: ?lanes.Sampling) !u32 {
    if (c.free()) return draw(row, position, s);
    var ids: [sampling.max_candidates]u32 = undefined;
    var vals: [sampling.max_candidates]f32 = undefined;
    const sm = s orelse return greedyAllowed(c, row);
    if (sm.temperature <= 0) return greedyAllowed(c, row);
    if (sm.top_k == 0) return error.NeedsTopK; // as sampleRow: the nucleus path without top_k
    const k = @min(sm.top_k, sampling.max_candidates);
    const n = c.top(row, k, ids[0..k], vals[0..k]);
    if (n == 0) return draw(row, position, s); // an empty row: nothing to constrain
    var wide: [sampling.max_candidates]i64 = undefined;
    for (wide[0..n], ids[0..n]) |*w, id| w.* = id;
    return @intCast(sampling.chooseAmong(vals[0..n], wide[0..n], position, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p }));
}

fn greedyAllowed(c: *grammar.Constraint, row: []const f32) u32 {
    var id: [1]u32 = undefined;
    var val: [1]f32 = undefined;
    if (c.top(row, 1, &id, &val) == 0) return @intCast(sampling.argmax(row)); // an empty row: nothing to constrain
    return id[0];
}

/// Where rank 0's backend calls spend their time (ns, summed; --profile): a round's frame to rank 1, its forward
/// and absorb (synchronized apart), the logits' copy to the host and the draws; the drafter passes; the prefills.
pub const Profile = struct {
    send: u64 = 0,
    enqueue: u64 = 0,
    forward: u64 = 0,
    absorb: u64 = 0,
    logits: u64 = 0,
    sample: u64 = 0,
    pass: u64 = 0,
    prefill: u64 = 0,
    rounds: u64 = 0,
    rows: u64 = 0,
    passes: u64 = 0,
    pass_streams: u64 = 0,
    prefills: u64 = 0,
};

/// A round-cost table "30.0,35.2,..." (TF_DS_ROUND_MS's form): ms by rows from 1.
pub fn parseMs(a: std.mem.Allocator, text: []const u8) ![]const f64 {
    return draft.scheduler_costs.parseRounds(a, text);
}

/// A stream's place in the pool and the drafts it holds.
const Lane = struct {
    slot: usize,
    base: usize,
    end: usize,
    first: u64 = 0, // the prompt's draw (a handle)
    held: [draft.max_block]u32 = undefined,
    confs: [draft.max_block]f32 = undefined,
    nheld: usize = 0,
    grammar: ?*grammar.Constraint = null, // the reply's structure (response_format, a required call)
    grammar_rows: u64 = 0, // rows drawn under it, and the host time those draws took (ns)
    grammar_ns: u64 = 0,
    tel: Telemetry = .{},
};

/// A request's rounds as the reply's telemetry (engine_api Stats.telemetry_json, the reply's speculative.proposer):
/// per verify width its rounds' count, ms sum and sum of squares; the drafter passes' the same; the drafts each round
/// verified, the accepted prefix, the committed tokens (the prefix and the drawn token after it); rounds whose first
/// draft was rejected, whose every draft was accepted, whose drafts held an end-of-sequence token, whose window
/// reached the reply's token budget; and the graphs' stretches captured, replayed and run eagerly meanwhile.
const Telemetry = struct {
    const Moments = struct {
        n: u64 = 0,
        sum: f64 = 0,
        sq: f64 = 0,
        fn add(x: *Moments, ms: f64) void {
            x.n += 1;
            x.sum += ms;
            x.sq += ms * ms;
        }
    };
    by_rows: [max_rows + 1]Moments = @splat(.{}),
    draft: Moments = .{},
    drafted: u64 = 0,
    accepted: u64 = 0,
    committed: u64 = 0,
    first_rejected: u64 = 0,
    full_accepted: u64 = 0,
    eos_drafted: u64 = 0,
    budget_reached: u64 = 0,
    graphs0: [3]u64 = .{ 0, 0, 0 }, // the graphs' counters at the request's prefill

    /// One verify window of a stream (its end-of-sequence tokens, the reply tokens it had, its budget).
    fn verified(t: *Telemetry, eos: []const u32, emitted: usize, max_new: usize, ids: []const i32, drawn: []const u32, ms: f64) void {
        const rows = ids.len;
        t.by_rows[@min(rows, max_rows)].add(ms);
        var kept: usize = 0; // the drafts accepted: rows whose drawn token is the next row's id
        while (kept + 1 < rows and drawn[kept] == @as(u32, @intCast(ids[kept + 1]))) kept += 1;
        t.drafted += rows - 1;
        t.accepted += kept;
        t.committed += kept + 1;
        if (rows > 1 and kept == 0) t.first_rejected += 1;
        if (rows > 1 and kept == rows - 1) t.full_accepted += 1;
        for (ids[1..]) |d| if (std.mem.indexOfScalar(u32, eos, @intCast(d)) != null) {
            t.eos_drafted += 1;
            break;
        };
        if (emitted + rows >= max_new) t.budget_reached += 1;
    }

    /// The JSON object into `buf` (empty when it does not fit).
    fn write(t: *const Telemetry, graphs: ?*const round.Graphs, buf: []u8) usize {
        var w: std.Io.Writer = .fixed(buf);
        t.json(graphs, &w) catch return 0;
        return w.end;
    }

    fn json(t: *const Telemetry, graphs: ?*const round.Graphs, w: *std.Io.Writer) !void {
        try w.writeAll("{\"rounds_by_rows\": {");
        var first = true;
        for (t.by_rows, 0..) |x, r| if (x.n > 0) {
            try w.print("{s}\"{d}\": [{d}, {d:.3}, {d:.3}]", .{ if (first) "" else ", ", r, x.n, x.sum, x.sq });
            first = false;
        };
        try w.print("}}, \"draft_ms\": [{d}, {d:.3}, {d:.3}], \"drafted\": {d}, \"accepted\": {d}, \"committed\": {d}, \"first_rejected\": {d}, \"full_accepted\": {d}, \"eos_drafted\": {d}, \"budget_reached\": {d}", .{ t.draft.n, t.draft.sum, t.draft.sq, t.drafted, t.accepted, t.committed, t.first_rejected, t.full_accepted, t.eos_drafted, t.budget_reached });
        if (graphs) |g| try w.print(", \"graphs\": {{\"captured\": {d}, \"replayed\": {d}, \"eager\": {d}}}", .{ g.captured - t.graphs0[0], g.replayed - t.graphs0[1], g.eager - t.graphs0[2] });
        try w.writeAll("}");
    }
};

extern "c" fn exp(x: f64) f64;

/// A kept prompt state (the prompt cache's saved state, kept at a chunk end the prompt pass reached before its
/// replay): the slot's own part on the device (Model.snapCopy), and the extent its compressed rows stay in,
/// [base, base + at), held back from placement while it lives. Stale: a placement needed that room (its restore fails
/// and the prompt runs from 0).
pub const Snap = struct {
    id: u64,
    at: usize,
    base: usize,
    buf: gpu.DeviceBuffer,
    stale: bool = false,
    wanted: u64 = 0, // when a request first waited for the live stream that holds these rows (placeKept; 0: none)
    end: usize = 0, // the extent end of the stream that kept it (placement holds [base, end) back while room allows)
};

/// How long a request waits for the live stream that holds the kept state it resumes from to end (then it resumes
/// in place and every kept state stays) before other kept states give their room for a copy of it.
const held_wait_ns: u64 = 10 * std.time.ns_per_s;

/// First tokens a handle names (each prompt's draw, read back right after its prefill).
const ring = 1024;

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    control: ?coll.Control = null,
    peer: ?link.Link, // rank 1 (null: one rank); every follower gets each frame (Model.followers)
    vision: ?*VisionTower = null, // rank 0's tower (a vision checkpoint served with its kit's vision/; CUDA builds only)
    step: u64 = 0,
    wire: Wire = .{},
    streams: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    wc: [window_rows]lanes.config.Cost = undefined, // the window costs (init)
    sc: [max_rows]lanes.config.Cost = undefined, // the shared forwards' costs (init)
    used: [max_streams]bool = @splat(false),
    seqs: [max_streams]std.ArrayList(i32) = @splat(.empty),
    built: Built = .{},
    prof: ?Profile = null,
    fix_k: ?usize = null, // a measurement (tf-dsv41-lanes --fix-k): K drafts a stream, chances 1 to K and 0 past it
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    snaps: std.ArrayList(*Snap) = .empty, // kept prompt states, least recently used first (made, or last resumed)
    next_snap: u64 = 1,
    /// The tokenizer's bytes for structured output (native.zig loads it); null: requests with a structure refused.
    vocab: ?*const grammar.Vocab = null,

    pub fn init(gpa: std.mem.Allocator, m: *Model) Lanes {
        var ln: Lanes = .{ .gpa = gpa, .m = m, .peer = if (m.world > 1) m.peer() else null };
        if (ln.peer != null) ln.control = m.coll.control;
        // the round costs by rows: ROUND_MS[min(rows, len) - 1] (the served table, or TF_DS_ROUND_MS)
        const t = m.round_ms;
        for (&ln.wc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        for (&ln.sc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        return ln;
    }

    /// Tells rank 1 to stop, then frees the host state (the model is the caller's).
    pub fn deinit(self: *Lanes) void {
        for (self.snaps.items) |sn| {
            sn.buf.free();
            self.gpa.destroy(sn);
        }
        self.snaps.deinit(self.gpa);
        if (self.peer != null) self.send(.done) catch {};
        self.streams.deinit(self.gpa);
        for (&self.seqs) |*q| q.deinit(self.gpa);
        self.wire.deinit(self.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .probabilities = probabilitiesFn,
            .release = releaseFn,
        } };
    }

    /// The facts the round loop reads: windows of the pending row and up to the drafter's block, several streams'
    /// windows in one forward of at most 16 rows (four streams), the drafter's chances, the served lane's round ms.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.m.drafting();
        return .{
            .exact_width = window_rows,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = if (self.m.dr) |dr| @intCast(dr.n) else 1,
            .window_costs = self.wc[0..],
            .mtp_step_ms = self.m.draft_ms,
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = @intCast(self.m.eng.round_rows),
            .max_streams = @intCast(self.m.streams),
            .shared_costs = self.sc[0..self.m.eng.round_rows],
            .draft_probabilities = drafting,
            .draft_streams = drafting,
        };
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The frame being built goes to rank 1 (after `begin` and the fields), or to ranks 1-3 on the four-node split; the
    /// step counts it only once it went out, so a frame that failed to build or send leaves rank 0 at rank 1's step.
    fn flush(self: *Lanes) !void {
        if (self.peer == null) return;
        if (self.control) |c| {
            try c.send(c.ptr, self.wire.buf.items);
        } else {
            for (self.m.followers()) |l| try l.?.send(self.wire.buf.items);
        }
        self.step += 1;
    }

    /// A frame of `kind` at the next step (its fields follow, then flush); false with no follower.
    fn begin(self: *Lanes, kind: Kind) !bool {
        if (self.peer == null) return false;
        try self.wire.begin(self.gpa, kind, self.step);
        return true;
    }

    fn send(self: *Lanes, kind: Kind) !void {
        if (try self.begin(kind)) try self.flush();
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// The lowest free slot, the first extent that fits, the prompt in chunks of the served size (at multiples of it:
    /// the chunks the gates checked; the host's chunk plan is not followed), the head's draw of the first token.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        return prefill(ptr, s) catch |err| {
            ended(err);
            return err;
        };
    }

    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        const t0 = m.now();
        defer if (self.prof) |*p| {
            p.prefill += m.now() - t0;
            p.prefills += 1;
        };
        const ids = s.prompt();
        const len = ids.len;
        if (len == 0) return error.EmptyPrompt;
        if (len + s.max_new > usableContext(m.pool_cap, m.eng.round_rows)) return error.PromptTooLong;
        const size = std.mem.alignForward(usize, len + s.max_new + m.eng.round_rows + 2, model.chunk_rows);
        if (self.streams.fetchRemove(s)) |old| {
            self.used[old.value.slot] = false;
            if (old.value.grammar) |c| c.deinit();
        }
        const slot = std.mem.indexOfScalar(bool, self.used[0..self.m.streams], false) orelse return error.NoFreeSlot;
        // a reply with a structure draws within its grammar (the server refused any the compiler does not take)
        var constraint: ?*grammar.Constraint = null;
        if (s.structure) |spec| {
            var diag: grammar.Diag = .{};
            constraint = grammar.Constraint.init(gpa, self.vocab orelse return error.NoStructures, spec, s.eos, &diag) catch |e| switch (e) {
                error.Unsupported => {
                    std.log.err("structured output: {s}", .{diag.text()});
                    return error.BadStructure;
                },
                else => return e,
            };
        }
        errdefer if (constraint) |c| c.deinit();
        var taken: [max_streams][2]usize = undefined;
        var nt: usize = 0;
        var it = self.streams.valueIterator();
        while (it.next()) |l| {
            taken[nt] = .{ l.base, l.end };
            nt += 1;
        }
        // a kept state of this prompt's prefix: the pass resumes at its chunk end, in its extent, or in one of its own
        // with the kept rows copied there when a live stream holds that extent (a burst's second request on a prompt)
        s.cached = 0;
        var kept: ?*Snap = null;
        var moved = false;
        var why: []const u8 = "no kept state"; // (the pass line's reason for running from 0)
        if (s.reuse.saved) |p| {
            const sn: *Snap = @ptrCast(@alignCast(p));
            // (a kept state knows no images: a prompt with them runs whole and keeps nothing)
            if (s.images.len == 0 and self.usable(sn, len)) {
                kept = sn;
                // in place unless the extent runs past the pool, a live stream holds it, or another kept state has
                // rows past the kept ones (resuming there would stale it: the rows go to a free extent instead, as
                // the Python lane's _place resumes "here" only in a free extent)
                moved = sn.base + size > m.pool_cap or clashes(taken[0..nt], sn.base, sn.base + size) or self.keptOver(sn.base + sn.at, sn.base + size, sn);
            } else {
                s.reuse_failed = true;
                why = "its kept state is not usable";
            }
        }
        const base = if (kept != null and !moved) kept.?.base else try self.placeKept(taken[0..nt], size, kept);
        if (kept) |sn| {
            sn.wanted = 0;
            if (!moved) sn.end = @max(sn.end, base + size); // its extent now: the next resume in place finds it held
        }
        if (kept) |sn| if (moved and sn.stale) { // the placement needed the kept rows' room: the prompt runs from 0
            kept = null;
            s.reuse_failed = true;
            why = "its kept rows went to this placement";
        };
        if (kept) |sn| self.touch(sn);
        // the rows this stream writes are no other kept state's
        self.staleOver(if (kept != null and !moved) base + kept.?.at else base, base + size, kept);
        if (self.peer != null) {
            var rb: [192]u8 = undefined;
            const detail: []const u8 = (if (kept) |sn| (if (moved)
                std.fmt.bufPrint(&rb, ", {d} rows; resumes state {d} at {d}, moved from {d}; {d} other stream(s) live", .{ size, sn.id, sn.at, sn.base, nt })
            else
                std.fmt.bufPrint(&rb, ", {d} rows; resumes state {d} at {d} in place; {d} other stream(s) live", .{ size, sn.id, sn.at, nt }))
            else
                std.fmt.bufPrint(&rb, ", {d} rows; from 0: {s}; {d} other stream(s) live", .{ size, why, nt })) catch "";
            notePass(m, self.step, 0, len, slot, base, detail);
        }
        var tel: Telemetry = .{};
        if (m.graphs) |*g| tel.graphs0 = .{ g.captured, g.replayed, g.eager };
        try self.streams.put(gpa, s, .{ .slot = slot, .base = base, .end = base + size, .grammar = constraint, .tel = tel });
        self.used[slot] = true;
        errdefer {
            _ = self.streams.remove(s);
            self.used[slot] = false;
        }
        const seq = &self.seqs[slot];
        try seq.resize(gpa, len);
        for (seq.items, ids) |*q, id| q.* = @intCast(id);
        // images: each span's positions dead in the sequence (negative: no Engram n-gram reaches into or across one),
        // its rows ready (a gate's) or the tower's (rank 0), all of them sent to rank 1 after the fill's start
        var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
        defer spans.deinit(gpa);
        var rows: std.ArrayList(u8) = .empty;
        defer rows.deinit(gpa);
        if (s.images.len > 0) {
            const image_id = m.cfg.image_token orelse return error.NoVision;
            const row_bytes = m.cfg.hidden * 2;
            var at_row: usize = 0;
            for (s.images) |img| {
                if (img.tokens == 0 or @as(usize, img.at) + img.tokens > len) return error.BadImageSpan;
                for (seq.items[img.at..][0..img.tokens]) |*q| {
                    if (q.* != @as(i64, image_id)) return error.BadImageSpan;
                    q.* = -1;
                }
                if (img.rows) {
                    if (img.bytes.len != img.tokens * row_bytes) return error.BadImageRows;
                    try rows.appendSlice(gpa, img.bytes);
                } else {
                    // a prepared picture (vision.zig): the tower's span rows, here on rank 0
                    if (!gpu.features.vision) return error.NoVisionTower;
                    const v = self.vision orelse return error.NoVisionTower;
                    if (try v.rows(img.bytes, gpa, &rows) != img.tokens) return error.BadImageSpan;
                }
                try spans.append(gpa, .{ .at = img.at, .len = img.tokens, .row = at_row });
                at_row += img.tokens;
            }
        }
        if (try self.begin(.fill_begin)) {
            try self.wire.int(gpa, u32, @intCast(slot));
            try self.wire.int(gpa, u64, base);
            try self.wire.ids(gpa, seq.items);
            try self.flush();
        }
        try m.fillBegin(slot, base);
        if (spans.items.len > 0) {
            // the spans and their rows, image_piece bytes of rows a frame (a follower's frame buffer holds 64 KiB and
            // the ids of a full pool)
            var off: usize = 0;
            while (off < rows.items.len) {
                const n = @min(image_piece, rows.items.len - off);
                if (try self.begin(.fill_images)) {
                    try self.wire.int(gpa, u32, @intCast(spans.items.len));
                    for (spans.items) |sp| {
                        try self.wire.int(gpa, u64, sp.at);
                        try self.wire.int(gpa, u64, sp.len);
                    }
                    try self.wire.int(gpa, u64, off);
                    try self.wire.buf.appendSlice(gpa, rows.items[off..][0..n]);
                    try self.flush();
                }
                off += n;
            }
            try m.setImages(spans.items, rows.items);
        }
        const replay = len -| m.cfg.window;
        var start: usize = 0;
        if (kept) |sn| {
            if (try self.begin(.snap_restore)) {
                try self.wire.int(gpa, u64, sn.id);
                try self.wire.int(gpa, u32, @intCast(slot));
                try self.wire.int(gpa, u64, sn.base);
                try self.wire.int(gpa, u64, base);
                try self.wire.int(gpa, u64, sn.at);
                try self.flush();
            }
            if (base != sn.base) try m.copyExtent(sn.base, base, sn.at);
            try m.snapCopy(slot, sn.buf.ptr, false);
            start = sn.at;
            s.cached = @intCast(sn.at);
        }
        // where this pass keeps states: the cache's marks and its last chunk end before the replay (the served lane's
        // kept prompt), each at the chunk end at or before it
        const last_end = @min(replay, len - 1) / model.chunk_rows * model.chunk_rows;
        var head = false;
        m.fill_seq = seq.items;
        defer {
            m.joinAhead();
            m.fill_seq = &.{};
        }
        while (start < len) {
            if (start > 0 and s.isCancelled()) return error.Cancelled;
            const n = @min(model.chunk_rows, len - start);
            const f: ChunkFrame = .{ .start = start, .n = n, .replay = replay, .pre_read = m.preRead(start, n) };
            if (try self.begin(.fill_chunk)) {
                try f.write(&self.wire, gpa);
                try self.flush();
            }
            head = try m.fillChunk(seq.items[0 .. start + n], start, n, replay, f.pre_read);
            start += n;
            if (s.images.len == 0) if (s.reuse.hook) |k| if (start < len and start <= replay and (start == last_end or markEnds(s.reuse.marks, start))) k.at(k.ptr, s, @intCast(start));
        }
        if (!head) return error.NoPromptLogits;
        const logits = try m.promptLogitsHost();
        if (logits_sha or logits_dump != null) {
            var dg: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(logits), &dg, .{});
            const hex = std.fmt.bytesToHex(dg, .lower);
            if (logits_sha) std.debug.print("{{\"prompt_logits\": \"{s}\", \"sha256\": \"{s}\", \"rows\": {d}}}\n", .{ s.id, hex, len });
            if (logits_dump) |dir| try dumpLogits(m.io, dir, s.id, len, m.cfg.vocab, logits, &hex);
        }
        const l = self.streams.getPtr(s).?;
        const token = if (l.grammar) |c| blk: {
            const g0 = m.now();
            defer {
                l.grammar_rows += 1;
                l.grammar_ns += m.now() - g0;
            }
            c.reset(s.emitted());
            break :blk try drawAllowed(c, logits, len, s.sampling);
        } else try draw(logits, len, s.sampling);
        l.first = self.take(token);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        return l.first;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = .{ ptr, s, feed, position };
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Every window as one round at its stream's cache length: the pending token, the held drafts (or the host's), each
    /// row drawn at its keyed position.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        return verify(ptr, windows, out) catch |err| {
            ended(err);
            return err;
        };
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (windows.len > max_streams) return error.WindowTooWide;
        var ids: [max_streams][max_rows]i32 = undefined;
        var wins: [max_streams]Win = undefined;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = self.streams.getPtr(w.stream) orelse return error.UnknownStream;
            const rows = w.rows();
            if (rows > window_rows) return error.WindowTooWide;
            const start: usize = @intCast(w.stream.cache_len);
            for (w.positions, 0..) |p, r| if (p != start + 1 + r) return error.PositionMismatch;
            if (w.held > l.nheld) return error.HeldMismatch;
            ids[k][0] = @intCast(w.pending);
            for (l.held[0..w.held], 0..) |t, j| ids[k][1 + j] = @intCast(t);
            for (w.tokens, 0..) |t, j| ids[k][1 + w.held + j] = @intCast(t);
            wins[k] = .{ .slot = l.slot, .base = l.base, .end = l.end, .start = start, .absorb = m.drafting() and w.stream.drafts, .ids = ids[k][0..rows] };
        }
        const t0 = m.now();
        if (try self.begin(.verify)) {
            try self.wire.int(gpa, u32, @intCast(windows.len));
            for (wins[0..windows.len]) |w| {
                try self.wire.int(gpa, u32, @intCast(w.slot));
                try self.wire.int(gpa, u64, w.base);
                try self.wire.int(gpa, u64, w.end);
                try self.wire.int(gpa, u64, w.start);
                try self.wire.int(gpa, u8, @intFromBool(w.absorb));
                try self.wire.ids(gpa, w.ids);
            }
            try self.flush();
        }
        const t1 = m.now();
        const rows = try self.built.build(gpa, &self.seqs, wins[0..windows.len]);
        try m.verify(rows, self.built.absorb[0..windows.len]);
        const t2 = m.now();
        // every row greedy: the device's argmax (R tokens back); else the logits for the host's draws
        var greedy = true;
        for (windows) |w| {
            if (w.stream.sampling) |sm| if (sm.temperature > 0) {
                greedy = false;
            };
        }
        const tokens: []const u32 = if (greedy) try m.roundArgmax(rows.ids.len) else &.{};
        const logits: []const f32 = if (greedy) &.{} else try m.roundLogits(rows.ids.len);
        const traced: []const f32 = if (trace_rounds >= 2) (if (greedy) try m.roundLogits(rows.ids.len) else logits) else &.{};
        const t3 = m.now();
        defer if (self.prof) |*p| {
            p.send += t1 - t0;
            p.enqueue += m.t_enqueue;
            p.forward += m.t_forward;
            p.absorb += m.t_absorb;
            p.logits += t3 - t2;
            p.sample += m.now() - t3;
            p.rounds += 1;
            p.rows += rows.ids.len;
        };
        const V = m.vocab;
        for (windows, out, 0..) |w, *o, k| {
            const row0 = self.built.wins[k].row;
            const l = self.streams.getPtr(w.stream).?;
            if (l.grammar) |c| {
                const g0 = m.now();
                try self.drawWindow(c, w, ids[k][0..w.rows()], tokens, logits, row0, o.sampled);
                l.grammar_rows += w.rows();
                l.grammar_ns += m.now() - g0;
            } else for (0..w.rows()) |r| o.sampled[r] = if (greedy) tokens[row0 + r] else try draw(logits[(row0 + r) * V ..][0..V], w.positions[r], w.stream.sampling);
            @memcpy(o.drafts[0..w.held], l.held[0..w.held]);
            @memcpy(o.drafts[w.held..][0..w.tokens.len], w.tokens);
            l.nheld = 0;
            l.tel.verified(w.stream.eos, w.stream.context.items.len - w.stream.prompt_len, w.stream.max_new, ids[k][0..w.rows()], o.sampled[0..w.rows()], @as(f64, @floatFromInt(m.now() - t0)) / 1e6);
            if (trace_rounds >= 2) {
                std.debug.print("{{\"verify\": {d}, \"rows\": {d}, \"ids\": {any}, \"drawn\": {any}, \"top2\": [", .{ w.stream.cache_len, w.rows(), ids[k][0..w.rows()], o.sampled[0..w.rows()] });
                for (0..w.rows()) |r| {
                    const t = topTwo(traced[(row0 + r) * V ..][0..V]);
                    std.debug.print("{s}[{d}, {e}, {d}, {e}]", .{ if (r == 0) "" else ", ", t[0][0], t[0][1], t[1][0], t[1][1] });
                }
                std.debug.print("]}}\n", .{});
            } else if (trace_rounds == 1) std.debug.print("{{\"verify\": {d}, \"rows\": {d}, \"ids\": {any}, \"drawn\": {any}}}\n", .{ w.stream.cache_len, w.rows(), ids[k][0..w.rows()], o.sampled[0..w.rows()] });
        }
    }

    /// A constrained window's draws: row r among the tokens the grammar allows after the reply and the window's rows
    /// 1..r (a row past a draft it refuses draws free: the round never keeps it, as a mismatch). Greedy rows (no
    /// `logits`) keep the device's argmax when the grammar allows it and read their logits row only when it does not.
    fn drawWindow(self: *Lanes, c: *grammar.Constraint, w: be.Window, row_ids: []const i32, tokens: []const u32, logits: []const f32, row0: usize, sampled: []u32) !void {
        const m = self.m;
        const V = m.vocab;
        c.reset(w.stream.emitted());
        var alive = true;
        for (0..w.rows()) |r| {
            if (r > 0 and alive) alive = c.feed(@intCast(row_ids[r]));
            const free = !alive or c.free();
            if (logits.len == 0) {
                const t = tokens[row0 + r];
                sampled[r] = if (free or c.allows(t)) t else greedyAllowed(c, try m.roundLogitsRow(row0 + r));
            } else {
                const row = logits[(row0 + r) * V ..][0..V];
                sampled[r] = if (free) try draw(row, w.positions[r], w.stream.sampling) else try drawAllowed(c, row, w.positions[r], w.stream.sampling);
            }
        }
    }

    /// The caches are positional: the next window starts at the stream's new length and writes over the rows past it.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        _ = ptr;
        for (windows, paths) |_, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        }
    }

    /// One drafter pass for every stream asked: its pending token at its new length, `depth` drafts held.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        return draftWindows(ptr, requests) catch |err| {
            ended(err);
            return err;
        };
    }

    fn draftWindows(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (m.dr == null) return error.NoDraftHead;
        const block = m.dr.?.n;
        var tokens: [max_streams]i64 = undefined;
        var q0: [max_streams]i64 = undefined;
        var slots: [max_streams]i64 = undefined;
        var depths: [max_streams]usize = undefined;
        var asked: [max_streams]*Lane = undefined;
        var n: usize = 0;
        var steps: usize = 0;
        for (requests) |r| {
            if (r.lanes != null or r.early) return error.TreesNotBuilt;
            const l = self.streams.getPtr(r.stream) orelse return error.UnknownStream;
            l.nheld = 0;
            if (r.depth == 0) continue;
            if (n == max_streams) return error.TooManyStreams;
            if (r.position != r.stream.cache_len + 1) return error.PositionMismatch;
            const token: u32 = if (r.rows == null) self.value(r.first orelse return error.NoFirstToken) else (if (r.follow.len > 0) r.follow[r.follow.len - 1] else return error.NoFollowToken);
            tokens[n] = token;
            q0[n] = @intCast(r.position - 1);
            slots[n] = @intCast(l.slot);
            depths[n] = @min(r.depth, block);
            asked[n] = l;
            steps = @max(steps, depths[n]);
            n += 1;
        }
        if (n == 0) return;
        if (try self.begin(.pass)) {
            try self.wire.int(gpa, u32, @intCast(n));
            try self.wire.int(gpa, u32, @intCast(steps));
            for (0..n) |i| {
                try self.wire.int(gpa, i64, tokens[i]);
                try self.wire.int(gpa, i64, q0[i]);
                try self.wire.int(gpa, i64, slots[i]);
            }
            try self.flush();
        }
        const t0 = m.now();
        try m.pass(tokens[0..n], q0[0..n], slots[0..n], steps);
        const pass_ms = @as(f64, @floatFromInt(m.now() - t0)) / 1e6;
        for (asked[0..n]) |l| l.tel.draft.add(pass_ms);
        if (self.prof) |*p| {
            p.pass += m.now() - t0;
            p.passes += 1;
            p.pass_streams += n;
        }
        const d = &m.dr.?;
        for (asked[0..n], depths[0..n], 0..) |l, k, i| {
            for (0..k) |j| {
                l.held[j] = @intCast(d.drafts[i][j]);
                l.confs[j] = d.confs[i][j];
            }
            l.nheld = k;
        }
    }

    /// The chance that each held draft lands with every earlier one, as the round loop's allocation reads it: the
    /// running product of the confidences' sigmoids (multi.py's conf rule: a sigmoid is the draft's survival once
    /// the earlier drafts survived, and the prefix survivals' sum is the expected kept drafts; draft.chooseK).
    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return false;
        if (out.len > l.nheld) return false;
        if (self.fix_k) |k| {
            for (out, 0..) |*o, j| o.* = if (j < k) 1.0 else 0.0;
            return true;
        }
        survivals(l.confs[0..out.len], out);
        return true;
    }

    /// Whether a kept state still restores for a prompt of `len` tokens: live, not stale, a chunk end before the
    /// prompt's replay (the rows before it ran the encoder layers alone, as a fresh pass of this prompt runs them).
    fn usable(self: *const Lanes, sn: *const Snap, len: usize) bool {
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, @constCast(sn)) == null) return false;
        return !sn.stale and sn.at > 0 and sn.at < len and sn.at % model.chunk_rows == 0 and sn.at <= len -| self.m.cfg.window;
    }

    /// The first extent that fits around the live streams and the kept states. Kept states give their room least
    /// recently used first (made, or last resumed: `touch`) and `source` (the state this prompt resumes from, its rows
    /// copied here) last, as the Python lane's _place takes the oldest other kept prompt before the source (oldest made
    /// first until 2026-10-22: a chat resumed a moment ago then lost its rows to agents' newer states while their
    /// older ones, never resumed, stayed); a state whose rows lie in a live stream's extent gives
    /// none (that stream holds the room) and stays. When a live stream holds `source`, the request first waits up to
    /// held_wait_ns for that stream to end (error.ContextFull: the host tries it again; it then resumes in place) if
    /// the copy needs another state's room or would leave none for another request of its size: a copy that fills the
    /// pool makes the next prompt evict a kept state (the 5 x 500K check with the duplicate admitted fourth). Kept
    /// states hold their stream's whole extent back while a placement still fits so (placeSoft): its tail stays free
    /// for the resume in place, which then stales nothing.
    fn placeKept(self: *Lanes, live: []const [2]usize, size: usize, source: ?*Snap) !usize {
        const gpa = self.gpa;
        var kept: std.ArrayList([3]usize) = .empty;
        defer kept.deinit(gpa);
        var rows: std.ArrayList([2]usize) = .empty;
        defer rows.deinit(gpa);
        const held = if (source) |sn| !sn.stale and clashes(live, sn.base, sn.base + sn.at) else false;
        while (true) {
            kept.clearRetainingCapacity();
            rows.clearRetainingCapacity();
            try rows.appendSlice(gpa, live);
            for (self.snaps.items) |sn| if (!sn.stale) {
                try kept.append(gpa, .{ sn.base, sn.at, sn.end });
                try rows.append(gpa, .{ sn.base, sn.base + sn.at });
            };
            const got = try placeSoft(gpa, live, kept.items, size, self.m.pool_cap);
            if (held and self.patient(source.?)) {
                const b = got orelse return error.ContextFull;
                return if (try roomAfter(gpa, rows.items, b, size, self.m.pool_cap)) b else error.ContextFull;
            }
            if (got) |b| return b;
            const victim = for (self.snaps.items) |sn| {
                if (!sn.stale and sn != source and !clashes(live, sn.base, sn.base + sn.at)) break sn;
            } else if (source) |sn| (if (!sn.stale and !clashes(live, sn.base, sn.base + sn.at)) sn else return error.ContextFull) else return error.ContextFull;
            victim.stale = true;
        }
    }

    /// A kept state a pass resumes from goes last in placement's order (placeKept takes the first), so kept states give
    /// their room least recently used first.
    fn touch(self: *Lanes, sn: *Snap) void {
        const i = std.mem.indexOfScalar(*Snap, self.snaps.items, sn) orelse return;
        _ = self.snaps.orderedRemove(i);
        self.snaps.appendAssumeCapacity(sn);
    }

    /// Whether a kept state other than `except` and its siblings (the same stream's other marks) has rows in [lo, hi).
    fn keptOver(self: *const Lanes, lo: usize, hi: usize, except: *const Snap) bool {
        for (self.snaps.items) |sn| if (sn != except and !sn.stale and sn.base != except.base and sn.base < hi and lo < sn.base + sn.at) return true;
        return false;
    }

    /// Whether a request still waits for the live stream that holds `sn` (held_wait_ns from its first wait).
    fn patient(self: *const Lanes, sn: *Snap) bool {
        const t = self.m.now();
        if (sn.wanted == 0) sn.wanted = t;
        return t - sn.wanted < held_wait_ns;
    }

    /// Kept states with rows in [lo, hi) go stale (a stream writes there), `except` aside.
    fn staleOver(self: *Lanes, lo: usize, hi: usize, except: ?*Snap) void {
        for (self.snaps.items) |sn| if (sn != except and sn.base < hi and lo < sn.base + sn.at) {
            sn.stale = true;
        };
    }

    /// The prompt cache's Snapshots functions (native.zig hands them to the CUDA host).
    pub fn snapBytesFn(ptr: *anyopaque, at: u32) u64 {
        _ = at;
        return of(ptr).m.snapBytes();
    }

    /// The live state at `at` (the pass stands at that chunk end): the slot's own part copied, the extent's rows kept.
    pub fn snapSaveFn(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        return snapSave(ptr, owner, at) catch |err| {
            ended(err);
            return err;
        };
    }

    fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = of(ptr);
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (at == 0 or at % model.chunk_rows != 0 or at > s.prompt_len -| self.m.cfg.window) return error.NotAChunkEnd;
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .id = self.next_snap, .at = at, .base = l.base, .end = l.end, .buf = try gpu.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()) };
        errdefer sn.buf.free();
        try self.snaps.append(self.gpa, sn);
        errdefer _ = self.snaps.pop(); // (the errdefers above free it: the list must not keep it)
        self.next_snap += 1;
        if (try self.begin(.snap_save)) {
            try self.wire.int(self.gpa, u64, sn.id);
            try self.wire.int(self.gpa, u32, @intCast(l.slot));
            try self.flush();
        }
        self.m.snapCopy(l.slot, sn.buf.ptr, true) catch |err| {
            // rank 1 has the frame and may hold its part under this id: it lets it go as this rank does (the errdefers)
            self.sendDrop(sn.id) catch {};
            return err;
        };
        return sn;
    }

    pub fn snapRestoreFn(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    pub fn snapDropFn(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        const i = std.mem.indexOfScalar(*Snap, self.snaps.items, sn) orelse return;
        _ = self.snaps.orderedRemove(i);
        self.sendDrop(sn.id) catch |err| {
            std.log.warn("dsv41: rank 1 keeps a dropped prompt state: {s}", .{@errorName(err)});
            ended(err); // a lost link is the pair's failure; the rank-1 copy of a state never sent is only memory
        };
        sn.buf.free();
        self.gpa.destroy(sn);
    }

    /// snap_drop to rank 1 (whole or not at all, as every frame: flush counts the step).
    fn sendDrop(self: *Lanes, id: u64) !void {
        if (!try self.begin(.snap_drop)) return;
        try self.wire.int(self.gpa, u64, id);
        try self.flush();
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.streams.fetchRemove(s) orelse return;
        self.used[kv.value.slot] = false;
        s.telemetry_len = kv.value.tel.write(if (self.m.graphs) |*g| g else null, &s.telemetry_buf);
        if (kv.value.grammar) |c| {
            const l = kv.value;
            std.log.info("structured reply: {d} rows drawn under its grammar, {d:.1} us a row on the host ({d} token checks, {d} full masks)", .{ l.grammar_rows, @as(f64, @floatFromInt(l.grammar_ns)) / 1e3 / @as(f64, @floatFromInt(@max(l.grammar_rows, 1))), c.checks, c.masks });
            c.deinit();
        }
    }
};

/// A fill_chunk frame's fields: the chunk's rows and replay, and rank 0's preRead, which orders a split chunk's Engram
/// gathers (Model.fillChunk) the same on every rank.
const ChunkFrame = struct {
    start: usize,
    n: usize,
    replay: usize,
    pre_read: bool,

    fn write(f: ChunkFrame, w: *Wire, gpa: std.mem.Allocator) !void {
        try w.int(gpa, u64, f.start);
        try w.int(gpa, u32, @intCast(f.n));
        try w.int(gpa, u64, f.replay);
        try w.int(gpa, u8, @intFromBool(f.pre_read));
    }

    fn read(r: *Reader) !ChunkFrame {
        return .{ .start = @intCast(try r.int(u64)), .n = try r.int(u32), .replay = @intCast(try r.int(u64)), .pre_read = switch (try r.int(u8)) {
            0 => false,
            1 => true,
            else => return error.BadFrame,
        } };
    }
};

/// A repeated warning's occurrences that are logged: the first, then every 100th.
fn oftenEnough(n: u64) bool {
    return n == 1 or (n != 0 and n % 100 == 0);
}

/// Whether a frame leaves the prompt pass in progress on rank 1 as it is (its next chunk's Engram read-ahead and ids):
/// the pass's own chunks and images, and the prompt cache's saves, restores and drops, which rank 0 sends between a
/// pass's chunks (a drop there, the failed restore's state reported at the first keep, once reset rank 1's plan while
/// rank 0 kept its own: the ranks then issued a split chunk's Engram gathers in different orders, 22 Oct 2026).
fn inPass(kind: Kind) bool {
    return switch (kind) {
        .fill_chunk, .fill_images, .snap_save, .snap_restore, .snap_drop => true,
        .fill_begin, .verify, .pass, .done => false,
    };
}

/// One line per prompt pass on each rank (no prompt text): the frame step (the ranks' common key), where it fills, what
/// it resumes from, the streams live, and the collectives' totals as the pass begins. With both streams idle the
/// totals are equal on the two ranks unless they fell out of step before this pass.
fn notePass(m: *Model, step: u64, rank: u32, len: usize, slot: usize, base: usize, detail: []const u8) void {
    const n = m.coll.counts();
    const idle = m.stream.done() catch false;
    std.log.info("lanes: step {d} rank {d} prompt pass: {d} tokens, slot {d}, extent at {d}{s}; collectives (stream {s}): staged {d}, cable1 {d}, cable2 {d}, host {d}", .{
        step, rank, len, slot, base, detail, if (idle) "idle" else "busy", n.staged, n.cable1, n.cable2, n.host,
    });
}

/// Prefix survivals: out[j] = the product of sigmoid(conf[i]) for i <= j, in float64 (libm's exp, as chooseK).
pub fn survivals(conf: []const f32, out: []f64) void {
    var surv: f64 = 1.0;
    for (out, conf[0..out.len]) |*o, cf| {
        surv *= 1.0 / (1.0 + exp(-@as(f64, cf)));
        o.* = surv;
    }
}

/// A primitive this rank refused: rank 0 runs the same one on the same inputs and refuses it at the same step (its
/// request ends with the error), so the follower reports it and takes the next frame instead of ending; unless the
/// pair itself failed (broken), which ends the follower (U47).
fn refused(err: anyerror) !void {
    std.debug.print("{{\"follow_refused\": \"{s}\"}}\n", .{@errorName(err)});
    if (broken(err)) return err;
}

/// A failure of the TP pair itself, after which neither rank can run another primitive: the collectives poisoned
/// (coll.Error), the control link gone, the ranks out of step.
fn broken(err: anyerror) bool {
    return switch (err) {
        error.CollectiveFailed, error.PeerLost, error.Timeout, error.Mismatch, error.Aborted, error.PeerDown, error.OutOfStep, error.FrameTooLong => true,
        else => false,
    };
}

/// Rank 0's side of U47: when a primitive fails because the pair is broken, the request in flight still gets its error,
/// then the process exits (status 3) half a second later, freeing its GPU memory and telling the launcher; rank 1 sees
/// the control link close (or its own collective fail) and exits too. Called from the backend's errdefers.
fn ended(err: anyerror) void {
    if (!broken(err) or exiting.swap(true, .acq_rel)) return;
    std.log.err("dsv41: the TP pair failed ({s}); rank 0 exits", .{@errorName(err)});
    const t = std.Thread.spawn(.{}, exitSoon, .{}) catch std.process.exit(3);
    t.detach();
}

var exiting: std.atomic.Value(bool) = .init(false);

fn exitSoon() void {
    const ts: std.c.timespec = .{ .sec = 0, .nsec = 500 * std.time.ns_per_ms };
    _ = std.c.nanosleep(&ts, null);
    std.process.exit(3);
}

/// A follower's verify times with Model.prof (a profile on ranks 1-3: rounds, issue and wall ns summed).
pub var follow_prof: struct { rounds: u64 = 0, enqueue: u64 = 0, forward: u64 = 0 } = .{};

/// Rank 1 (each of ranks 1-3 on the four-node split): the primitives rank 0 sends, run in order until it says done.
pub fn follow(gpa: std.mem.Allocator, m: *Model) !void {
    const peer = m.peer();
    const control = m.coll.control;
    const buf = try gpa.alloc(u8, coll.controlFrameBytes(m.pool_cap));
    defer gpa.free(buf);
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    var win_ids: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&win_ids) |*q| q.deinit(gpa);
    var built: Built = .{};
    var step: u64 = 0;
    var slot: usize = 0;
    var plan_diffs: u64 = 0; // chunks whose read-ahead plan here differed from rank 0's (the warning: oftenEnough)
    var img_rows: std.ArrayList(u8) = .empty; // a prompt's image rows as their pieces come
    defer img_rows.deinit(gpa);
    var kept: std.AutoHashMapUnmanaged(u64, gpu.DeviceBuffer) = .empty; // rank 0's kept states, this rank's part
    defer {
        var ki = kept.valueIterator();
        while (ki.next()) |b| b.free();
        kept.deinit(gpa);
    }
    while (true) {
        // the next primitive, polled for 20 ms before the read blocks (rank 0 sends it a fraction of a millisecond
        // after this rank's GPU goes idle in a decode; a sleeping thread would add its core's wake-up to every frame)
        const spin_ns = 20 * std.time.ns_per_ms;
        const frame = if (control) |c| try c.recv(c.ptr, buf, spin_ns) else try peer.recvSpin(buf, spin_ns);
        var r: Reader = .{ .b = frame };
        const kind: Kind = switch (try r.int(u8)) {
            1 => .fill_begin,
            2 => .fill_chunk,
            3 => .verify,
            4 => .pass,
            5 => .snap_save,
            6 => .snap_restore,
            7 => .snap_drop,
            8 => .fill_images,
            9 => .done,
            else => return error.BadFrame,
        };
        if (try r.int(u64) != step) return error.OutOfStep;
        step += 1;
        // a chunk's read-ahead of the next is done before anything that ends the prompt pass (its ids may change)
        if (!inPass(kind)) {
            m.joinAhead();
            m.fill_seq = &.{};
        }
        switch (kind) {
            .fill_begin => {
                slot = try r.int(u32);
                const base = try r.int(u64);
                if (slot >= max_streams) return error.BadSlot;
                img_rows.clearRetainingCapacity();
                try r.ids(gpa, &seqs[slot]);
                try r.end();
                notePass(m, step - 1, 1, seqs[slot].items.len, slot, @intCast(base), "");
                m.fillBegin(slot, @intCast(base)) catch |err| try refused(err);
                m.fill_seq = seqs[slot].items;
            },
            .fill_chunk => {
                const f = try ChunkFrame.read(&r);
                try r.end();
                if (f.start + f.n > seqs[slot].items.len) return error.BadChunk;
                if (f.pre_read != m.preRead(f.start, f.n)) {
                    plan_diffs += 1;
                    if (oftenEnough(plan_diffs)) std.log.warn("lanes: step {d} rank 1: its Engram read-ahead plan for the chunk at {d} differs from rank 0's ({}); rank 0's orders the gathers ({d} such chunks so far)", .{ step - 1, f.start, f.pre_read, plan_diffs });
                }
                _ = m.fillChunk(seqs[slot].items[0 .. f.start + f.n], f.start, f.n, f.replay, f.pre_read) catch |err| try refused(err);
            },
            .verify => {
                const nw = try r.int(u32);
                if (nw == 0 or nw > max_streams) return error.BadFrame;
                var wins: [max_streams]Win = undefined;
                for (0..nw) |k| {
                    const s: usize = try r.int(u32);
                    const base: usize = @intCast(try r.int(u64));
                    const end: usize = @intCast(try r.int(u64));
                    const start: usize = @intCast(try r.int(u64));
                    const absorb = try r.int(u8) != 0;
                    try r.ids(gpa, &win_ids[k]);
                    wins[k] = .{ .slot = s, .base = base, .end = end, .start = start, .absorb = absorb, .ids = win_ids[k].items };
                }
                try r.end();
                const rows = try built.build(gpa, &seqs, wins[0..nw]);
                m.verify(rows, built.absorb[0..nw]) catch |err| try refused(err);
                if (m.prof) {
                    follow_prof.rounds += 1;
                    follow_prof.enqueue += m.t_enqueue;
                    follow_prof.forward += m.t_forward;
                }
            },
            .pass => {
                const n = try r.int(u32);
                const steps = try r.int(u32);
                if (n == 0 or n > max_streams) return error.BadFrame;
                var tokens: [max_streams]i64 = undefined;
                var q0: [max_streams]i64 = undefined;
                var slots: [max_streams]i64 = undefined;
                for (0..n) |i| {
                    tokens[i] = try r.int(i64);
                    q0[i] = try r.int(i64);
                    slots[i] = try r.int(i64);
                }
                try r.end();
                m.pass(tokens[0..n], q0[0..n], slots[0..n], steps) catch |err| try refused(err);
            },
            .fill_images => {
                const count = try r.int(u32);
                var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
                defer spans.deinit(gpa);
                var row: usize = 0;
                for (0..count) |_| {
                    const at: usize = @intCast(try r.int(u64));
                    const n: usize = @intCast(try r.int(u64));
                    try spans.append(gpa, .{ .at = at, .len = n, .row = row });
                    row += n;
                }
                const off: usize = @intCast(try r.int(u64));
                const piece = r.rest();
                try r.end();
                const total = row * m.cfg.hidden * 2;
                if (off != img_rows.items.len or off + piece.len > total or piece.len == 0) return error.BadFrame;
                try img_rows.appendSlice(gpa, piece);
                if (img_rows.items.len == total) {
                    m.setImages(spans.items, img_rows.items) catch |err| try refused(err);
                    img_rows.clearRetainingCapacity();
                }
            },
            .snap_save => {
                const id = try r.int(u64);
                const s: usize = try r.int(u32);
                try r.end();
                if (s >= max_streams) return error.BadSlot;
                var b = try gpu.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
                m.snapCopy(s, b.ptr, true) catch |err| {
                    b.free();
                    try refused(err);
                    continue;
                };
                try kept.put(gpa, id, b);
            },
            .snap_restore => {
                const id = try r.int(u64);
                const s: usize = try r.int(u32);
                if (s >= max_streams) return error.BadSlot;
                const from: usize = @intCast(try r.int(u64));
                const to: usize = @intCast(try r.int(u64));
                const at: usize = @intCast(try r.int(u64));
                try r.end();
                const b = kept.get(id) orelse return error.UnknownSnapshot; // rank 0 restores only states both ranks keep
                std.log.info("lanes: step {d} rank 1 restores state {d} at {d} into slot {d}, extent {d} from {d}", .{ step - 1, id, at, s, to, from });
                if (from != to) m.copyExtent(from, to, at) catch |err| try refused(err);
                m.snapCopy(s, b.ptr, false) catch |err| try refused(err);
            },
            .snap_drop => {
                const id = try r.int(u64);
                try r.end();
                if (kept.fetchRemove(id)) |kv| {
                    var b = kv.value;
                    b.free();
                }
            },
            .done => {
                try r.end();
                return;
            },
        }
    }
}

test "draft chances are the prefix survivals chooseK sums" {
    var out: [3]f64 = undefined;
    survivals(&.{ 0.0, 0.0, 2.0 }, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), out[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.25 / (1.0 + @exp(@as(f64, -2.0))), out[2], 1e-12);
}

test "extents take the first chunk-aligned gap" {
    const used = [_][2]usize{ .{ 0, 4096 }, .{ 6144, 8192 } };
    try std.testing.expectEqual(@as(?usize, 4096), place(&used, 2048, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 8192), place(&used, 4096, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 0), place(&.{}, 2048, 2048));
    try std.testing.expectEqual(@as(?usize, null), place(&used, 4096, 10240));
}

test "a frame's fields read back as written" {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    try w.begin(gpa, .verify, 7);
    try w.int(gpa, u64, 1 << 40);
    try w.ids(gpa, &.{ 1, -2, 129279 });
    var r: Reader = .{ .b = w.buf.items };
    try std.testing.expectEqual(@as(u8, @backingInt(Kind.verify)), try r.int(u8));
    try std.testing.expectEqual(@as(u64, 7), try r.int(u64));
    try std.testing.expectEqual(@as(u64, 1 << 40), try r.int(u64));
    var got: std.ArrayList(i32) = .empty;
    defer got.deinit(gpa);
    try r.ids(gpa, &got);
    try std.testing.expectEqualSlices(i32, &.{ 1, -2, 129279 }, got.items);
    try std.testing.expectError(error.ShortFrame, r.int(u8));
}

test "a round's rows: each window at its positions, its ids written over the rejected rows" {
    const gpa = std.testing.allocator;
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    try seqs[0].appendSlice(gpa, &.{ 10, 11, 12, 13, 99, 98 }); // positions 4.. hold a rejected window
    try seqs[2].appendSlice(gpa, &.{ 20, 21 });
    var b: Built = .{};
    const rows = try b.build(gpa, &seqs, &.{
        .{ .slot = 0, .base = 0, .end = 4096, .start = 4, .absorb = true, .ids = &.{ 14, 15 } },
        .{ .slot = 2, .base = 4096, .end = 8192, .start = 2, .absorb = false, .ids = &.{22} },
    });
    try std.testing.expectEqualSlices(i64, &.{ 14, 15, 22 }, rows.ids);
    try std.testing.expectEqualSlices(i64, &.{ 4, 5, 2 }, rows.pos);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 2 }, rows.slots.?);
    try std.testing.expectEqualSlices(i64, &.{ 4096, 4096, 8192 }, rows.ends.?);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 12, 13, 14, 15 }, rows.windows.?[0].seq);
    try std.testing.expectEqualSlices(i32, &.{ 20, 21, 22 }, rows.windows.?[1].seq);
    try std.testing.expectEqual(@as(usize, 2), rows.windows.?[1].row);
    try std.testing.expectError(error.SequenceGap, b.build(gpa, &seqs, &.{.{ .slot = 1, .base = 0, .end = 2048, .start = 3, .absorb = false, .ids = &.{1} }}));
}

test "greedy draws take the first largest logit" {
    const row = [_]f32{ 0.5, 2.0, -1.0, 2.0 };
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, null));
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, .{ .seed = 3, .temperature = 0 }));
}

test "retained histories preserve their replay and output reservations (PR #10: eight 100K histories)" {
    var m: Model = undefined;
    m.pool_cap = 1 << 20;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    var snapshots: [8]Snap = undefined;
    for (&snapshots, 0..) |*sn, i| {
        const base = try ln.placeKept(&.{}, 102400, null);
        try std.testing.expectEqual(i * 102400, base);
        sn.* = .{ .id = i, .at = 98304, .base = base, .end = base + 102400, .buf = undefined };
        try ln.snaps.append(std.testing.allocator, sn);
    }
    // All eight 100K histories can resume within their reservations without invalidating a neighbor.
    for (&snapshots) |*sn| {
        try std.testing.expect(!ln.keptOver(sn.base + sn.at, sn.end, sn));
        ln.staleOver(sn.base + sn.at, sn.end, sn);
    }
    for (snapshots) |sn| try std.testing.expect(!sn.stale);
    // A larger continuation relocates into free room instead of overwriting the next cached prefix.
    try std.testing.expect(ln.keptOver(snapshots[0].at, 104448, &snapshots[0]));
    try std.testing.expectEqual(@as(usize, 819200), try ln.placeKept(&.{}, 104448, null));
    // At genuine capacity pressure, kept states may still be evicted; a live range never is.
    const live = [_][2]usize{.{ 0, 102400 }};
    try std.testing.expectEqual(@as(usize, 102400), try ln.placeKept(&live, 409600, null));
    try std.testing.expect(ln.snaps.items[1].stale and ln.snaps.items[4].stale);
    try std.testing.expect(!ln.snaps.items[5].stale);
    try std.testing.expect(!ln.snaps.items[0].stale); // its rows lie in the live stream: that stream holds the room
}

test "kept states give their room least recently used first: a resumed one goes last" {
    var m: Model = undefined;
    m.pool_cap = 3 * 8192;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    var sn: [3]Snap = undefined;
    for (&sn, 0..) |*x, i| {
        x.* = .{ .id = i + 1, .at = 8192, .base = i * 8192, .end = (i + 1) * 8192, .buf = undefined };
        try ln.snaps.append(std.testing.allocator, x);
    }
    // the pool is full: a placement takes a kept state's room, the least recently used one's
    ln.touch(&sn[0]); // the oldest made was just resumed
    try std.testing.expectEqual(@as(usize, 8192), try ln.placeKept(&.{}, 8192, null));
    try std.testing.expect(!sn[0].stale and sn[1].stale and !sn[2].stale);
    try std.testing.expectEqual(@as(u64, 1), ln.snaps.items[ln.snaps.items.len - 1].id);
    // then the next least recently used (sn[2]; sn[1] already gave its room)
    try std.testing.expectEqual(@as(usize, 16384), try ln.placeKept(&.{.{ 8192, 16384 }}, 8192, null));
    try std.testing.expect(!sn[0].stale and sn[2].stale);
    var gone: Snap = .{ .id = 9, .at = 8192, .base = 0, .end = 8192, .buf = undefined };
    ln.touch(&gone); // a state no longer kept: nothing moves
    try std.testing.expectEqual(@as(usize, 3), ln.snaps.items.len);
    try std.testing.expectEqual(@as(u64, 1), ln.snaps.items[2].id);
}

test "a dropped state's frame goes out whole or not at all, and only a sent frame counts a step" {
    const gpa = std.testing.allocator;
    var sv: [2]std.c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &sv));
    defer for (sv) |fd| {
        _ = std.c.close(fd);
    };
    var m: Model = undefined;
    m.world = 2;
    m.fan = .{};
    m.fan.links[0] = .{ .fd = sv[0] };
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = m.fan.links[0], .step = 7 };
    defer ln.wire.deinit(gpa);
    // a frame begun but not sent counts no step; the next one carries the same step
    try std.testing.expect(try ln.begin(.verify));
    try std.testing.expectEqual(@as(u64, 7), ln.step);
    try ln.sendDrop(42);
    try std.testing.expectEqual(@as(u64, 8), ln.step);
    var buf: [64]u8 = undefined;
    const peer: link.Link = .{ .fd = sv[1] };
    var r: Reader = .{ .b = try peer.recvSpin(&buf, 0) };
    try std.testing.expectEqual(@as(u8, @backingInt(Kind.snap_drop)), try r.int(u8));
    try std.testing.expectEqual(@as(u64, 7), try r.int(u64));
    try std.testing.expectEqual(@as(u64, 42), try r.int(u64));
    // a frame that cannot be built is not sent and counts no step
    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    var ln2: Lanes = .{ .gpa = failing.allocator(), .m = &m, .peer = m.fan.links[0], .step = 8 };
    defer ln2.wire.deinit(failing.allocator());
    try std.testing.expectError(error.OutOfMemory, ln2.sendDrop(43));
    try std.testing.expectEqual(@as(u64, 8), ln2.step);
    var one: [1]u8 = undefined;
    try std.testing.expect(std.c.recv(sv[1], &one, 1, std.c.MSG.DONTWAIT) < 0); // nothing arrived
}

test "a frame is read to its end or refused; a repeated warning is logged first and every 100th" {
    const bytes = [_]u8{ 1, 2, 3 };
    var r: Reader = .{ .b = &bytes };
    _ = try r.int(u16);
    try std.testing.expectError(error.BadFrame, r.end()); // a field more than this build reads
    _ = try r.int(u8);
    try r.end();
    for ([_]u64{ 1, 100, 200 }) |n| try std.testing.expect(oftenEnough(n));
    for ([_]u64{ 0, 2, 99, 101 }) |n| try std.testing.expect(!oftenEnough(n));
}

test "the cache's frames between a pass's chunks leave rank 1's pass as it is; the others end it" {
    for ([_]Kind{ .fill_chunk, .fill_images, .snap_save, .snap_restore, .snap_drop }) |k| try std.testing.expect(inPass(k));
    for ([_]Kind{ .fill_begin, .verify, .pass, .done }) |k| try std.testing.expect(!inPass(k));
}

test "a chunk's frame carries rank 0's read-ahead plan to rank 1" {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    for ([_]bool{ true, false }) |pre| {
        try w.begin(gpa, .fill_chunk, 9);
        const f: ChunkFrame = .{ .start = 16384, .n = 1348, .replay = 17604, .pre_read = pre };
        try f.write(&w, gpa);
        var r: Reader = .{ .b = w.buf.items };
        _ = try r.int(u8);
        _ = try r.int(u64);
        const g = try ChunkFrame.read(&r);
        try std.testing.expectEqual(f, g);
        try std.testing.expectEqual(r.b.len, r.at);
    }
    var bad: [21]u8 = @splat(0);
    bad[20] = 2;
    var rb: Reader = .{ .b = &bad };
    try std.testing.expectError(error.BadFrame, ChunkFrame.read(&rb));
}

test "a request's telemetry: accepted prefixes, first rejections, full acceptance, EOS drafts, the budget" {
    var t: Telemetry = .{};
    // 4 rows (3 drafts): two accepted, the third rejected
    t.verified(&.{9}, 0, 100, &.{ 5, 6, 7, 8 }, &.{ 6, 7, 1, 2 }, 40.0);
    // 3 rows: the first draft rejected; a draft is the end of sequence
    t.verified(&.{9}, 3, 100, &.{ 5, 9, 7 }, &.{ 4, 0, 0 }, 30.0);
    // 3 rows, every draft accepted, the window reaching the budget
    t.verified(&.{9}, 97, 100, &.{ 1, 2, 3 }, &.{ 2, 3, 4 }, 30.0);
    // a plain round
    t.verified(&.{9}, 99, 100, &.{1}, &.{2}, 20.0);
    try std.testing.expectEqual(@as(u64, 7), t.drafted);
    try std.testing.expectEqual(@as(u64, 4), t.accepted);
    try std.testing.expectEqual(@as(u64, 8), t.committed);
    try std.testing.expectEqual(@as(u64, 1), t.first_rejected);
    try std.testing.expectEqual(@as(u64, 1), t.full_accepted);
    try std.testing.expectEqual(@as(u64, 1), t.eos_drafted);
    try std.testing.expectEqual(@as(u64, 2), t.budget_reached);
    t.draft.add(6.0);
    var buf: [512]u8 = undefined;
    const n = t.write(null, &buf);
    try std.testing.expectEqualStrings("{\"rounds_by_rows\": {\"1\": [1, 20.000, 400.000], \"3\": [2, 60.000, 1800.000], \"4\": [1, 40.000, 1600.000]}, \"draft_ms\": [1, 6.000, 36.000], \"drafted\": 7, \"accepted\": 4, \"committed\": 8, \"first_rejected\": 1, \"full_accepted\": 1, \"eos_drafted\": 1, \"budget_reached\": 2}", buf[0..n]);
    // a buffer too small for the object: nothing, rather than a cut JSON
    var small: [16]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 0), t.write(null, &small));
}
