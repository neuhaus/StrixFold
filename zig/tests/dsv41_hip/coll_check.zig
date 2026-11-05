//! tf-dsv41-coll-check RANK MASTER_IP PORT [--rounds N] [--prefill-chunks N] [--receipt PATH] [--fault KIND --at N]
//!   [--fault-calls decode|prefill] [--kit DIR] [--graph N] [--corrupt R:N] [--soak SECONDS] [--prefill-stress N]
//!   [--overlap N [--work-mib M | --work-us U]]:
//! w-link's tcp_host
//! collectives (tcp_coll.zig) between the two boxes over the real USB4 link, through the family's own open path (the
//! control link, the peers' handshake, the data links), with device buffers on both ends:
//! - validation: every call of a DSpark decode round (80 x 120 KiB, 2 x 600 KiB, a 1.55 MB head, 6 x 100 KiB drafter
//!   gathers, 5 x 16 B Markov gathers), a plain decode round, a 16-row round (80 x 320 KiB, 2 x 1.6 MB, a 4.14 MB
//!   head) and 2048-row prefill chunks (40 MiB and 200 MiB gathers):
//!   both halves of every recv compared word for word with the patterns each rank wrote for that call;
//! - timing: the same rounds back to back without the validation copies (the host time of each all_gather call, and
//!   the round's wall time to its last upload).
//! Both ranks run the same sequence. Take the GPU/link lock and the quiet lock on both boxes (node0 first). Settings
//! come from the environment as in serving (TF_TP_STRIPE, TF_COLL_*).
//!
//! With TF_COLL_KIND=proxy or verbs give --kit (the dsv41 kit with coll.hip). --graph N then also captures one
//! DSpark pass (94 gathers, each into its own send and recv region) into a HIP graph and replays it N times, writing
//! fresh patterns before each replay and checking every word after it, then times N back-to-back replays.
//!
//! Fault mode (COLLECTIVES-DESIGN.md section 7): the rank given --fault runs DSpark rounds and, at call N, dies
//! (kill: SIGKILL itself), hangs (stall: stops calling for 2x the progress deadline, then exits), issues a different
//! site (site), calls abort from another thread while its call waits (abort), or closes the collective (close). Every
//! rank prints one line "fault-result: rank=R calls=K error=E after_ms=T" when its loop ends: the call count it
//! completed, the error that ended it (none if it finished) and how long that call took to fail. With
//! --fault-calls prefill the loop runs 2048-row prefill chunks (40 and 200 MiB gathers) instead of DSpark rounds, so an
//! external kill or SIGSTOP lands inside a large gather.
//!
//! Soak (--soak S, the same on both ranks): S seconds of mixed traffic. Each iteration validates one DSpark and one
//! plain round, times ten of each, and with --graph (any N) validates one replay of the captured DSpark pass and times
//! ten; every 50th iteration validates a 2048-row prefill chunk. The ranks agree to stop through a 16 B gather of
//! keep-going flags at the end of each iteration. Prints a progress line every 60 s and the distributions at the end.
//!
//! Prefill stress (--prefill-stress N, the same on both ranks): two validated 2048-row prefill chunks, then N chunks
//! back to back without the check copies (the RTO floor question: do 40/200 MiB gathers ever stall). Prints each
//! size's per-call host time, the chunk times, the five slowest chunks, and on a failure the chunk, the call and the
//! error (the collective is terminal after it). Exit 0 when all N complete.
//!
//! Overlap (--overlap N, the same on both ranks; coll.Coll.gatherBegin / gatherEnd): does a split gather move its bytes
//! while the GPU runs other work. "Work" on the gathering stream is either a device-to-device copy of M MiB
//! (--work-mib, default 512) between buffers no gather touches (memory-bound: it competes with the staging copies and
//! the network for DRAM), or with --work-us U a one-thread kernel that spins U microseconds on the wall clock (no memory
//! traffic): the two ends between which real compute sits. First every call of a 2048-row prefill chunk is validated
//! through begin / work / end. Then N repetitions of five prefill-chunk shapes, interleaved: "work" (one work per
//! half-chunk step, no gathers); "gather" (the chunk's gathers alone, as --prefill-stress); "halves" (each gather as two
//! half-size gathers, no work: what the split costs); "serial" (each step's two works, then its whole gather, as the
//! engine runs a chunk today); "overlap" (the chunk as two halves of 1024 rows: begin A's half-size gather, B's work,
//! end A, begin B, A's next work, end B; the same work and bytes). Prints each shape's chunk time and the share of the
//! smaller of work and gather that the overlap hid. Exit 0 when the validation finds no bad word.
//!
//! Negative control (--corrupt R:N, the same on both ranks): rank R flips bit 0 of word 0 of its send pattern at
//! validation call N, and with --graph at gather (N-1) mod 94 of the first validated replay. Every rank must then find
//! exactly that one bad word, in half R of that call, and nowhere else; it prints "negative-control: ... PASS|FAIL",
//! skips the timing passes and exits 0 on PASS, 4 on FAIL.

const std = @import("std");
const gpu = @import("gpu");
const dsv41 = @import("dsv41");
const coll = dsv41.coll;
/// tf_smoke_wait_write, --overlap's spin work
const smoke = @import("smoke_kernels");

const Call = struct { phase: coll.Phase, bytes: usize, count: usize };

const decode_round = [_]Call{
    .{ .phase = .round_attn, .bytes = 6 * 5120 * 4, .count = 80 },
    .{ .phase = .round_engram, .bytes = 6 * 25600 * 4, .count = 2 },
    .{ .phase = .round_head, .bytes = 6 * 64640 * 4, .count = 1 },
    .{ .phase = .draft_attn, .bytes = 5 * 5120 * 4, .count = 6 },
    .{ .phase = .draft_markov, .bytes = 16, .count = 5 },
};
const plain_round = [_]Call{
    .{ .phase = .round_attn, .bytes = 5120 * 4, .count = 80 },
    .{ .phase = .round_engram, .bytes = 25600 * 4, .count = 2 },
    .{ .phase = .round_head, .bytes = 64640 * 4, .count = 1 },
};
/// A 16-row verify round (several agents: C4 and C8 rounds fill the ROUND_ROWS cap), without the drafter.
const wide_round = [_]Call{
    .{ .phase = .round_attn, .bytes = 16 * 5120 * 4, .count = 80 },
    .{ .phase = .round_engram, .bytes = 16 * 25600 * 4, .count = 2 },
    .{ .phase = .round_head, .bytes = 16 * 64640 * 4, .count = 1 },
};
const prefill_chunk = [_]Call{
    .{ .phase = .prompt_engram, .bytes = 2048 * 25600 * 4, .count = 1 },
    .{ .phase = .prompt_attn, .bytes = 2048 * 5120 * 4, .count = 20 },
    .{ .phase = .prompt_engram, .bytes = 2048 * 25600 * 4, .count = 1 },
    .{ .phase = .prompt_moe, .bytes = 2048 * 5120 * 4, .count = 20 },
};

const Corrupt = struct { rank: u32, call: u64 };
var corrupt: ?Corrupt = null;
/// Where the checks found bad words (the first `hits.len` places), for the negative control.
const Hit = struct { graph: bool, call: u64, half: usize, words: usize };
var hits: [8]Hit = undefined;
var nhits: usize = 0;

fn hit(h: Hit) void {
    if (nhits < hits.len) hits[nhits] = h;
    nhits += 1;
}

fn passCalls(calls: []const Call) usize {
    var n: usize = 0;
    for (calls) |cl| n += cl.count;
    return n;
}

fn word(seq: u64, rank: u32, i: usize) u32 {
    var x = seq *% 0x9E3779B97F4A7C15 ^ (@as(u64, rank) << 56) ^ (@as(u64, i) *% 0xBF58476D1CE4E5B9);
    x ^= x >> 31;
    x *%= 0x94D049BB133111EB;
    x ^= x >> 29;
    return @truncate(x);
}

fn maxBytes(calls: []const Call) usize {
    var m: usize = 0;
    for (calls) |cl| m = @max(m, cl.bytes);
    return m;
}

const Env = struct {
    d: *const gpu.Driver,
    s: gpu.Stream,
    c: coll.Coll,
    rank: u32,
    send: gpu.DeviceBuffer,
    recv: gpu.DeviceBuffer,
    host: []u32,
    seq: u64 = 0,
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// One pass over `calls` with every payload written and checked; returns the number of bad words.
fn validate(e: *Env, calls: []const Call) !usize {
    var bad: usize = 0;
    var layer: usize = 0;
    for (calls) |cl| for (0..cl.count) |_| {
        e.seq += 1;
        const words = (cl.bytes + 3) / 4;
        for (e.host[0..words], 0..) |*w, i| w.* = word(e.seq, e.rank, i);
        if (corrupt) |c| {
            if (c.rank == e.rank and c.call == e.seq) e.host[0] ^= 1;
        }
        try gpu.copyH2D(e.d, try e.send.at(0), std.mem.sliceAsBytes(e.host[0..words])[0..cl.bytes]);
        try e.c.allGather(.at(cl.phase, layer), try e.send.at(0), try e.recv.at(0), cl.bytes, .u8, e.s);
        layer += 1;
        try e.s.synchronize();
        for (0..2) |r| {
            try gpu.copyD2H(e.d, std.mem.sliceAsBytes(e.host[0..words])[0..cl.bytes], try e.recv.at(r * cl.bytes));
            const full = cl.bytes / 4;
            var bh: usize = 0;
            for (e.host[0..full], 0..) |w, i| bh += @intFromBool(w != word(e.seq, @intCast(r), i));
            if (cl.bytes % 4 != 0) {
                const want = word(e.seq, @intCast(r), full);
                const got = std.mem.sliceAsBytes(e.host[0..words])[full * 4 .. cl.bytes];
                bh += @intFromBool(!std.mem.eql(u8, got, std.mem.asBytes(&want)[0..got.len]));
            }
            if (bh > 0) hit(.{ .graph = false, .call = e.seq, .half = r, .words = bh });
            bad += bh;
        }
    };
    return bad;
}

const Stats = struct {
    ns: std.ArrayList(u64) = .empty,

    fn print(st: *Stats, name: []const u8, out: *std.ArrayList(u8), gpa: std.mem.Allocator) !void {
        const v = st.ns.items;
        if (v.len == 0) return;
        std.mem.sort(u64, v, {}, std.sort.asc(u64));
        const p = struct {
            fn at(x: []const u64, q: f64) f64 {
                return @as(f64, @floatFromInt(x[@intFromFloat(q * @as(f64, @floatFromInt(x.len - 1)) + 0.5)])) / 1e3;
            }
        }.at;
        var sum: u64 = 0;
        for (v) |x| sum += x;
        const line = try std.fmt.allocPrint(gpa, "{s}: n={d} p50={d:.1} p95={d:.1} p99={d:.1} max={d:.1} mean={d:.1} us\n", .{ name, v.len, p(v, 0.5), p(v, 0.95), p(v, 0.99), @as(f64, @floatFromInt(v[v.len - 1])) / 1e3, @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(v.len)) / 1e3 });
        std.debug.print("{s}", .{line});
        try out.appendSlice(gpa, line);
    }
};

/// `rounds` passes back to back without validation copies: each call's host time and the round's wall time.
fn time(e: *Env, gpa: std.mem.Allocator, name: []const u8, calls: []const Call, rounds: usize, warm: usize, out: *std.ArrayList(u8)) !void {
    var per = std.AutoHashMap(usize, Stats).init(gpa);
    var round_ns: Stats = .{};
    for (0..warm + rounds) |r| {
        try e.s.synchronize();
        const t0 = nowNs();
        var layer: usize = 0;
        for (calls) |cl| for (0..cl.count) |_| {
            e.seq += 1;
            const a = nowNs();
            try e.c.allGather(.at(cl.phase, layer), try e.send.at(0), try e.recv.at(0), cl.bytes, .u8, e.s);
            const b = nowNs();
            layer += 1;
            if (r >= warm) {
                const g = try per.getOrPut(cl.bytes);
                if (!g.found_existing) g.value_ptr.* = .{};
                try g.value_ptr.ns.append(gpa, b - a);
            }
        };
        try e.s.synchronize();
        if (r >= warm) try round_ns.ns.append(gpa, nowNs() - t0);
    }
    for (calls) |cl| {
        if (per.getPtr(cl.bytes)) |st| {
            try st.print(try std.fmt.allocPrint(gpa, "{s} all_gather {d} B (host time per call)", .{ name, cl.bytes }), out, gpa);
            _ = per.remove(cl.bytes);
        }
    }
    try round_ns.print(try std.fmt.allocPrint(gpa, "{s} whole pass, wall to the last upload", .{name}), out, gpa);
}

const Fault = enum { kill, stall, site, abort, close };
var peer_mode = false;

const Aborter = struct {
    fn run(c: coll.Coll) void {
        var ts: std.c.timespec = .{ .sec = 0, .nsec = 200 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
        c.abort(.cancelled);
    }
};

/// DSpark rounds (or prefill chunks) until `at` calls, where this rank (with a fault) injects it; the peer
/// (--peer-faults) just runs until its collective fails. One result line either way.
fn faultRun(e: *Env, fault: ?Fault, at: usize, round: []const Call) !u8 {
    var calls: usize = 0;
    var failed: ?anyerror = null;
    var after_ns: u64 = 0;
    outer: for (0..1000) |_| {
        var layer: usize = 0;
        for (round) |cl| for (0..cl.count) |_| {
            if (fault) |f| if (calls == at) switch (f) {
                .kill => {
                    _ = std.c.raise(.KILL);
                },
                .stall => {
                    var ts: std.c.timespec = .{ .sec = 5, .nsec = 0 };
                    _ = std.c.nanosleep(&ts, null);
                    std.debug.print("fault-result: rank={d} calls={d} error=stalled after_ms=0\n", .{ e.rank, calls });
                    std.c._exit(0);
                },
                .close => {
                    e.c.close();
                    std.debug.print("fault-result: rank={d} calls={d} error=closed after_ms=0\n", .{ e.rank, calls });
                    std.c._exit(0);
                },
                .abort => {
                    const t = std.Thread.spawn(.{}, Aborter.run, .{e.c}) catch null;
                    if (t) |th| th.detach();
                    // wait without the peer (it keeps going): this call blocks until the abort lands
                },
                .site => {},
            };
            const phase: coll.Phase = if (fault == .site and calls == at) .test_only else cl.phase;
            const t0 = nowNs();
            e.c.allGather(.at(phase, layer), try e.send.at(0), try e.recv.at(0), cl.bytes, .u8, e.s) catch |err| {
                failed = err;
                after_ns = nowNs() - t0;
                break :outer;
            };
            layer += 1;
            calls += 1;
        };
        e.s.synchronize() catch |err| {
            failed = err;
            break :outer;
        };
    }
    std.debug.print("fault-result: rank={d} calls={d} error={s} after_ms={d}\n", .{ e.rank, calls, if (failed) |f| @errorName(f) else "none", after_ns / std.time.ns_per_ms });
    return if (failed == null) 0 else 3;
}

/// One DSpark pass captured into a HIP graph, each gather into its own send and recv regions.
const GraphPass = struct {
    calls: []Call,
    send_off: []usize,
    recv_off: []usize,
    gsend: gpu.DeviceBuffer,
    grecv: gpu.DeviceBuffer,
    g: gpu.graph.Graph,
    exec: gpu.graph.Exec,

    fn init(e: *Env, gpa: std.mem.Allocator) !GraphPass {
        var calls: std.ArrayList(Call) = .empty;
        for (decode_round) |cl| for (0..cl.count) |_| try calls.append(gpa, .{ .phase = cl.phase, .bytes = cl.bytes, .count = 1 });
        const send_off = try gpa.alloc(usize, calls.items.len);
        const recv_off = try gpa.alloc(usize, calls.items.len);
        var st: usize = 0;
        var rt: usize = 0;
        for (calls.items, 0..) |cl, k| {
            send_off[k] = st;
            recv_off[k] = rt;
            st += std.mem.alignForward(usize, cl.bytes, 256);
            rt += std.mem.alignForward(usize, 2 * cl.bytes, 256);
        }
        var gsend = try gpu.DeviceBuffer.alloc(e.d, st);
        errdefer gsend.free();
        var grecv = try gpu.DeviceBuffer.alloc(e.d, rt);
        errdefer grecv.free();
        try e.s.synchronize();
        try gpu.graph.beginCapture(e.s, .thread_local);
        for (calls.items, 0..) |cl, k|
            try e.c.allGather(.at(cl.phase, k), try gsend.at(send_off[k]), try grecv.at(recv_off[k]), cl.bytes, .u8, e.s);
        var g = try gpu.graph.endCapture(e.s);
        errdefer g.deinit();
        var exec = try g.instantiate();
        errdefer exec.deinit();
        try exec.upload(e.s);
        return .{ .calls = calls.items, .send_off = send_off, .recv_off = recv_off, .gsend = gsend, .grecv = grecv, .g = g, .exec = exec };
    }

    fn deinit(p: *GraphPass) void {
        p.exec.deinit();
        p.g.deinit();
        p.grecv.free();
        p.gsend.free();
    }

    /// Replay number `r` with fresh patterns, every word checked; returns the bad word count.
    fn check(p: *GraphPass, e: *Env, r: usize) !usize {
        for (p.calls, 0..) |cl, k| {
            const words = cl.bytes / 4;
            for (e.host[0..words], 0..) |*w, i| w.* = word(r * 1000 + k, e.rank, i);
            if (corrupt) |c| {
                if (r == 0 and c.rank == e.rank and k == (c.call - 1) % p.calls.len) e.host[0] ^= 1;
            }
            try gpu.copyH2D(e.d, try p.gsend.at(p.send_off[k]), std.mem.sliceAsBytes(e.host[0..words]));
        }
        try p.replay(e);
        var bad: usize = 0;
        for (p.calls, 0..) |cl, k| {
            const words = cl.bytes / 4;
            for (0..2) |rk| {
                try gpu.copyD2H(e.d, std.mem.sliceAsBytes(e.host[0..words]), try p.grecv.at(p.recv_off[k] + rk * cl.bytes));
                var bh: usize = 0;
                for (e.host[0..words], 0..) |w, i| bh += @intFromBool(w != word(r * 1000 + k, @intCast(rk), i));
                if (bh > 0) hit(.{ .graph = true, .call = r * p.calls.len + k + 1, .half = rk, .words = bh });
                bad += bh;
            }
        }
        return bad;
    }

    fn replay(p: *GraphPass, e: *Env) !void {
        try p.exec.launchOn(e.s);
        try e.s.synchronize();
    }
};

/// One DSpark pass captured into a HIP graph, replayed `reps` times with fresh patterns and every word checked, then
/// `reps` replays back to back timed. Returns the bad word count.
fn graphTest(e: *Env, gpa: std.mem.Allocator, reps: usize, out: *std.ArrayList(u8)) !usize {
    if (!e.c.capturable) {
        std.debug.print("graph test skipped: this collective is not capturable (use TF_COLL_KIND=proxy or verbs)\n", .{});
        return 0;
    }
    var p = try GraphPass.init(e, gpa);
    defer p.deinit();
    var bad: usize = 0;
    const t0 = nowNs();
    for (0..reps) |r| bad += try p.check(e, r);
    const vline = try std.fmt.allocPrint(gpa, "graph: {d} gathers captured, {d} validated replays, bad words {d}, {d} ms\n", .{ p.calls.len, reps, bad, (nowNs() - t0) / std.time.ns_per_ms });
    std.debug.print("{s}", .{vline});
    try out.appendSlice(gpa, vline);
    var per: Stats = .{};
    for (0..reps + 10) |r| {
        const a = nowNs();
        try p.replay(e);
        if (r >= 10) try per.ns.append(gpa, nowNs() - a);
    }
    try per.print("graph replay of one DSpark pass (94 gathers), launch to sync", out, gpa);
    return bad;
}

/// One pass over `calls` without validation copies; returns its wall time to the last upload.
fn pass(e: *Env, calls: []const Call) !u64 {
    try e.s.synchronize();
    const t0 = nowNs();
    var layer: usize = 0;
    for (calls) |cl| for (0..cl.count) |_| {
        e.seq += 1;
        try e.c.allGather(.at(cl.phase, layer), try e.send.at(0), try e.recv.at(0), cl.bytes, .u8, e.s);
        layer += 1;
    };
    try e.s.synchronize();
    return nowNs() - t0;
}

/// --prefill-stress: see the file comment. Returns the exit code.
fn prefillStress(e: *Env, gpa: std.mem.Allocator, n: usize, out: *std.ArrayList(u8)) !u8 {
    var bad: usize = 0;
    for (0..2) |_| bad += try validate(e, &prefill_chunk);
    const vline = try std.fmt.allocPrint(gpa, "prefill-stress: 2 validated chunks, bad words {d}\n", .{bad});
    std.debug.print("{s}", .{vline});
    try out.appendSlice(gpa, vline);
    var chunk_ns: Stats = .{};
    var big: Stats = .{};
    var small: Stats = .{};
    const Slow = struct { ns: u64, k: usize };
    var slow: [5]Slow = @splat(.{ .ns = 0, .k = 0 });
    var done: usize = 0;
    const t_all = nowNs();
    outer: for (0..n) |k| {
        try e.s.synchronize();
        const t0 = nowNs();
        var layer: usize = 0;
        for (prefill_chunk) |cl| for (0..cl.count) |_| {
            e.seq += 1;
            const a = nowNs();
            e.c.allGather(.at(cl.phase, layer), try e.send.at(0), try e.recv.at(0), cl.bytes, .u8, e.s) catch |err| {
                const fl = try std.fmt.allocPrint(gpa, "prefill-stress FAILED: chunk {d} call {d} ({d} B): {s} after {d} ms\n", .{ k, layer, cl.bytes, @errorName(err), (nowNs() - a) / std.time.ns_per_ms });
                std.debug.print("{s}", .{fl});
                try out.appendSlice(gpa, fl);
                break :outer;
            };
            const dt = nowNs() - a;
            try (if (cl.bytes > 64 << 20) &big else &small).ns.append(gpa, dt);
            layer += 1;
        };
        e.s.synchronize() catch |err| {
            const fl = try std.fmt.allocPrint(gpa, "prefill-stress FAILED: chunk {d} at its sync: {s}\n", .{ k, @errorName(err) });
            std.debug.print("{s}", .{fl});
            try out.appendSlice(gpa, fl);
            break :outer;
        };
        const dt = nowNs() - t0;
        try chunk_ns.ns.append(gpa, dt);
        var m: usize = 0;
        for (slow, 0..) |sl, j| {
            if (sl.ns < slow[m].ns) m = j;
        }
        if (dt > slow[m].ns) slow[m] = .{ .ns = dt, .k = k };
        done += 1;
    }
    const line = try std.fmt.allocPrint(gpa, "prefill-stress: {d} of {d} chunks completed in {d} ms ({d} gathers)\n", .{ done, n, (nowNs() - t_all) / std.time.ns_per_ms, done * passCalls(&prefill_chunk) });
    std.debug.print("{s}", .{line});
    try out.appendSlice(gpa, line);
    try big.print("prefill-stress all_gather 209715200 B (host time per call)", out, gpa);
    try small.print("prefill-stress all_gather 41943040 B (host time per call)", out, gpa);
    try chunk_ns.print("prefill-stress 2048-row chunk, wall to the last upload", out, gpa);
    std.mem.sort(Slow, &slow, {}, struct {
        fn gt(_: void, a: Slow, b: Slow) bool {
            return a.ns > b.ns;
        }
    }.gt);
    for (slow) |sl| if (sl.ns > 0) {
        const sline = try std.fmt.allocPrint(gpa, "prefill-stress slow chunk {d}: {d:.1} ms\n", .{ sl.k, @as(f64, @floatFromInt(sl.ns)) / 1e6 });
        std.debug.print("{s}", .{sline});
        try out.appendSlice(gpa, sline);
    };
    return if (done == n and bad == 0) 0 else 1;
}

/// --overlap's stand-in for compute on the gathering stream (see the file comment).
const Work = union(enum) {
    copy: struct { a: u64, b: u64, bytes: usize },
    spin: struct { f: gpu.Function, flag: u64, ticks: u64 },

    fn run(w: Work, e: *const Env) !void {
        switch (w) {
            .copy => |c| try gpu.copyD2DAsync(e.d, e.s.handle, c.b, c.a, c.bytes),
            .spin => |sp| {
                var a: gpu.Args = .{};
                a.add(sp.flag);
                a.add(@as(u32, 1));
                a.add(sp.ticks);
                try gpu.launch.launch(sp.f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, e.s, &a);
            },
        }
    }
};

/// One pass over `calls` like validate, each gather split (begin, work that does not touch its buffers, end).
fn validateSplit(e: *Env, calls: []const Call, work: Work) !usize {
    var bad: usize = 0;
    var layer: usize = 0;
    for (calls) |cl| for (0..cl.count) |_| {
        e.seq += 1;
        const words = (cl.bytes + 3) / 4;
        for (e.host[0..words], 0..) |*w, i| w.* = word(e.seq, e.rank, i);
        try gpu.copyH2D(e.d, try e.send.at(0), std.mem.sliceAsBytes(e.host[0..words])[0..cl.bytes]);
        const p = try e.c.gatherBegin(.at(cl.phase, layer), try e.send.at(0), try e.recv.at(0), cl.bytes, .u8, e.s);
        try work.run(e);
        try e.c.gatherEnd(p, e.s);
        layer += 1;
        try e.s.synchronize();
        for (0..2) |r| {
            try gpu.copyD2H(e.d, std.mem.sliceAsBytes(e.host[0..words])[0..cl.bytes], try e.recv.at(r * cl.bytes));
            var bh: usize = 0;
            for (e.host[0 .. cl.bytes / 4], 0..) |w, i| bh += @intFromBool(w != word(e.seq, @intCast(r), i));
            if (bh > 0) hit(.{ .graph = false, .call = e.seq, .half = r, .words = bh });
            bad += bh;
        }
    };
    return bad;
}

/// --overlap: see the file comment. Returns the exit code.
fn overlap(e: *Env, gpa: std.mem.Allocator, n: usize, work: Work, what: []const u8, out: *std.ArrayList(u8)) !u8 {
    const bad = try validateSplit(e, &prefill_chunk, work);
    const vline = try std.fmt.allocPrint(gpa, "overlap: validated {d} split gathers of a 2048-row chunk with {s} inside each, bad words {d}\n", .{ passCalls(&prefill_chunk), what, bad });
    std.debug.print("{s}", .{vline});
    try out.appendSlice(gpa, vline);
    const Shape = enum { work, gather, halves, serial, overlap };
    const shapes = comptime std.enums.values(Shape);
    var st: [shapes.len]Stats = @splat(.{});
    const send = try e.send.at(0);
    const recv = try e.recv.at(0);
    for (0..n) |_| {
        for (shapes, 0..) |shape, si| {
            try e.s.synchronize();
            const t0 = nowNs();
            var layer: usize = 0;
            if (shape != .gather and shape != .halves) try work.run(e); // A's first step
            for (prefill_chunk) |cl| for (0..cl.count) |_| {
                const site: coll.Site = .at(cl.phase, layer);
                const half = cl.bytes / 2;
                switch (shape) {
                    .work => for (0..2) |_| try work.run(e),
                    .gather => try e.c.allGather(site, send, recv, cl.bytes, .u8, e.s),
                    .serial => {
                        for (0..2) |_| try work.run(e);
                        try e.c.allGather(site, send, recv, cl.bytes, .u8, e.s);
                    },
                    .halves, .overlap => {
                        const pa = try e.c.gatherBegin(site, send, recv, half, .u8, e.s);
                        if (shape == .overlap) try work.run(e); // B's step while A's half moves
                        try e.c.gatherEnd(pa, e.s);
                        const pb = try e.c.gatherBegin(site, send + half, recv + cl.bytes, half, .u8, e.s);
                        if (shape == .overlap) try work.run(e); // A's next step while B's half moves
                        try e.c.gatherEnd(pb, e.s);
                    },
                }
                layer += 1;
            };
            try e.s.synchronize();
            try st[si].ns.append(gpa, nowNs() - t0);
        }
    }
    var p50: [shapes.len]f64 = undefined;
    for (&st, shapes, 0..) |*x, shape, si| {
        const name = try std.fmt.allocPrint(gpa, "overlap {s} (2048-row chunk, {s} per half-step)", .{ @tagName(shape), what });
        try x.print(name, out, gpa);
        p50[si] = @as(f64, @floatFromInt(x.ns.items[x.ns.items.len / 2])) / 1e6;
    }
    const w = p50[@intFromEnum(Shape.work)];
    const g = p50[@intFromEnum(Shape.gather)];
    const ser = p50[@intFromEnum(Shape.serial)];
    const ov = p50[@intFromEnum(Shape.overlap)];
    const line = try std.fmt.allocPrint(gpa, "overlap: p50 work {d:.1} ms, gather {d:.1} ms, halves {d:.1} ms, serial {d:.1} ms, overlap {d:.1} ms: saved {d:.1} ms = {d:.0}% of the smaller\n", .{ w, g, p50[@intFromEnum(Shape.halves)], ser, ov, ser - ov, 100 * (ser - ov) / @min(w, g) });
    std.debug.print("{s}", .{line});
    try out.appendSlice(gpa, line);
    return if (bad == 0) 0 else 1;
}

/// --soak: see the file comment. Returns the bad word count.
fn soak(e: *Env, gpa: std.mem.Allocator, seconds: u64, graphs: bool, out: *std.ArrayList(u8)) !usize {
    var gp: ?GraphPass = if (graphs and e.c.capturable) try GraphPass.init(e, gpa) else null;
    defer if (gp) |*p| p.deinit();
    var dspark: Stats = .{};
    var plain: Stats = .{};
    var replay: Stats = .{};
    var prefill: Stats = .{};
    var bad: usize = 0;
    var calls: usize = 0;
    var checked: usize = 0;
    const t0 = nowNs();
    var next_report = t0 + 60 * std.time.ns_per_s;
    var it: usize = 0;
    while (true) : (it += 1) {
        bad += try validate(e, &decode_round);
        bad += try validate(e, &plain_round);
        calls += passCalls(&decode_round) + passCalls(&plain_round);
        for (0..10) |_| {
            try dspark.ns.append(gpa, try pass(e, &decode_round));
            try plain.ns.append(gpa, try pass(e, &plain_round));
            calls += passCalls(&decode_round) + passCalls(&plain_round);
        }
        if (gp) |*p| {
            bad += try p.check(e, checked);
            checked += 1;
            for (0..10) |_| {
                const a = nowNs();
                try p.replay(e);
                try replay.ns.append(gpa, nowNs() - a);
            }
            calls += 11 * p.calls.len;
        }
        if (it % 50 == 49) {
            const a = nowNs();
            bad += try validate(e, &prefill_chunk);
            try prefill.ns.append(gpa, nowNs() - a);
            calls += passCalls(&prefill_chunk);
        }
        // both ranks stop together: a 16 B gather of keep-going flags
        const now = nowNs();
        e.host[0] = @intFromBool(now - t0 < seconds * std.time.ns_per_s);
        @memset(e.host[1..4], 0);
        try gpu.copyH2D(e.d, try e.send.at(0), std.mem.sliceAsBytes(e.host[0..4]));
        try e.c.allGather(.at(.test_only, 0), try e.send.at(0), try e.recv.at(0), 16, .u8, e.s);
        try e.s.synchronize();
        try gpu.copyD2H(e.d, std.mem.sliceAsBytes(e.host[0..8]), try e.recv.at(0));
        calls += 1;
        if (now >= next_report) {
            std.debug.print("soak: {d} s, {d} iterations, {d} calls, bad words {d}\n", .{ (now - t0) / std.time.ns_per_s, it + 1, calls, bad });
            next_report += 60 * std.time.ns_per_s;
        }
        if (e.host[0] == 0 or e.host[4] == 0) break;
    }
    const line = try std.fmt.allocPrint(gpa, "soak: {d} s, {d} iterations, {d} calls ({d} validated graph replays), bad words {d}\n", .{ (nowNs() - t0) / std.time.ns_per_s, it + 1, calls, checked, bad });
    std.debug.print("{s}", .{line});
    try out.appendSlice(gpa, line);
    try dspark.print("soak dspark whole pass, wall to the last upload", out, gpa);
    try plain.print("soak plain whole pass, wall to the last upload", out, gpa);
    try replay.print("soak graph replay of one DSpark pass, launch to sync", out, gpa);
    try prefill.print("soak validated 2048-row prefill chunk (with the check copies)", out, gpa);
    return bad;
}

fn checkControl(gpa: std.mem.Allocator, control: coll.Control, rank: u32, capacity: usize) !void {
    const buf = try gpa.alloc(u8, capacity);
    defer gpa.free(buf);
    const sizes = [_]usize{ 48, 256 << 10, capacity };
    for (0..6) |i| {
        const n = sizes[i % sizes.len];
        if (rank == 0) {
            for (buf[0..n], 0..) |*b, j| b.* = @truncate(j ^ (j >> 8) ^ i);
            try control.send(control.ptr, buf[0..n]);
        } else {
            const frame = try control.recv(control.ptr, buf, 20 * std.time.ns_per_ms);
            if (frame.len != n) return error.ControlFrameLength;
            for (frame, 0..) |b, j| {
                if (b != @as(u8, @truncate(j ^ (j >> 8) ^ i))) return error.ControlFrameCorrupt;
            }
        }
    }
    std.debug.print("verbs control: rank {d}, six frames up to {d} bytes checked\n", .{ rank, capacity });
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 4) {
        std.debug.print("usage: tf-dsv41-coll-check RANK MASTER_IP PORT [--rounds N] [--prefill-chunks N] [--receipt PATH]\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const master = try dsv41.link.parseIp(args[2]);
    const port = try std.fmt.parseInt(u16, args[3], 10);
    var rounds: usize = 200;
    var chunks: usize = 3;
    var receipt: ?[]const u8 = null;
    var fault: ?Fault = null;
    var fault_at: usize = 500;
    var kit_dir: ?[]const u8 = null;
    var graph_reps: usize = 0;
    var fault_calls: []const Call = &decode_round;
    var soak_s: u64 = 0;
    var stress_n: usize = 0;
    var overlap_n: usize = 0;
    var work_mib: usize = 512;
    var work_us: u64 = 0;
    var i: usize = 4;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--rounds")) rounds = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--prefill-chunks")) chunks = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--receipt")) receipt = args[i + 1];
        if (std.mem.eql(u8, args[i], "--fault")) fault = std.meta.stringToEnum(Fault, args[i + 1]) orelse return error.BadFault;
        if (std.mem.eql(u8, args[i], "--at")) fault_at = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--kit")) kit_dir = args[i + 1];
        if (std.mem.eql(u8, args[i], "--graph")) graph_reps = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--peer-faults")) peer_mode = std.mem.eql(u8, args[i + 1], "1");
        if (std.mem.eql(u8, args[i], "--fault-calls")) fault_calls = if (std.mem.eql(u8, args[i + 1], "prefill")) &prefill_chunk else if (std.mem.eql(u8, args[i + 1], "decode")) &decode_round else return error.BadFaultCalls;
        if (std.mem.eql(u8, args[i], "--soak")) soak_s = try std.fmt.parseInt(u64, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--prefill-stress")) stress_n = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--overlap")) overlap_n = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--work-mib")) work_mib = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--work-us")) work_us = try std.fmt.parseInt(u64, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--corrupt")) {
            const colon = std.mem.indexOfScalar(u8, args[i + 1], ':') orelse return error.BadCorrupt;
            corrupt = .{
                .rank = try std.fmt.parseInt(u32, args[i + 1][0..colon], 10),
                .call = try std.fmt.parseInt(u64, args[i + 1][colon + 1 ..], 10),
            };
            if (corrupt.?.rank > 1 or corrupt.?.call == 0) return error.BadCorrupt;
        }
    }
    var d = try gpu.Driver.open();
    defer d.close();
    var ctx = try gpu.Context.init(&d, 0);
    defer ctx.deinit();
    var s = try gpu.Stream.init(&d, true);
    defer s.deinit();
    var fan = try dsv41.fan.Fan.open(io, master, port, rank, 2);
    defer fan.close();
    const id: coll.Identity = .{ .rank = rank, .world = 2, .kit_abi = 0, .kit_sha256 = @splat('0'), .modalities = 0, .plan = "tool=coll_check", .control_bytes = coll.controlFrameBytes(1 << 20) };
    const caps = try ctx.caps();
    const kit: ?*gpu.Kit = if (kit_dir) |k| try gpu.Kit.open(gpa, io, &d, k, caps.archName(), .{}) else null;
    defer if (kit) |k| k.deinit();
    const c = try dsv41.tcp_coll.open(gpa, io, &ctx, kit, &fan, &id);
    defer c.close();
    if (c.control) |control| try checkControl(gpa, control, rank, id.control_bytes);
    const max = @max(maxBytes(&prefill_chunk), maxBytes(&decode_round));
    var send = try gpu.DeviceBuffer.alloc(&d, max);
    defer send.free();
    var recv = try gpu.DeviceBuffer.alloc(&d, 2 * max);
    defer recv.free();
    const host = try std.heap.page_allocator.alloc(u32, max / 4 + 1);
    defer std.heap.page_allocator.free(host);
    var e: Env = .{ .d = &d, .s = s, .c = c, .rank = rank, .send = send, .recv = recv, .host = host };
    if (fault != null or peer_mode) return faultRun(&e, fault, fault_at, fault_calls);
    var out: std.ArrayList(u8) = .empty;
    if (stress_n > 0) {
        const rc = try prefillStress(&e, gpa, stress_n, &out);
        if (receipt) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
        return rc;
    }
    if (overlap_n > 0) {
        var wbuf = try gpu.DeviceBuffer.alloc(&d, if (work_us > 0) 4096 else 2 * (work_mib << 20));
        defer wbuf.free();
        var m = try gpu.Module.load(&d, &smoke.image);
        defer m.unload();
        const work: Work = if (work_us > 0)
            .{ .spin = .{ .f = try m.function("tf_smoke_wait_write"), .flag = try wbuf.at(0), .ticks = work_us * @as(u64, @intCast(try ctx.attribute(.wall_clock_rate))) / 1000 } }
        else
            .{ .copy = .{ .a = try wbuf.at(0), .b = try wbuf.at(work_mib << 20), .bytes = work_mib << 20 } };
        const what = if (work_us > 0) try std.fmt.allocPrint(gpa, "a {d} us spin kernel", .{work_us}) else try std.fmt.allocPrint(gpa, "a {d} MiB copy", .{work_mib});
        const rc = try overlap(&e, gpa, overlap_n, work, what, &out);
        if (receipt) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
        return rc;
    }
    if (soak_s > 0) {
        const sbad = try soak(&e, gpa, soak_s, graph_reps > 0, &out);
        if (receipt) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
        return if (sbad == 0) 0 else 1;
    }
    var bad: usize = 0;
    var calls: usize = 0;
    const t0 = nowNs();
    for (0..10) |_| {
        bad += try validate(&e, &decode_round);
        bad += try validate(&e, &plain_round);
        bad += try validate(&e, &wide_round);
        calls += 94 + 83 + 83;
    }
    for (0..chunks) |_| {
        bad += try validate(&e, &prefill_chunk);
        calls += 42;
    }
    const line = try std.fmt.allocPrint(gpa, "validation: {d} calls (10 DSpark rounds, 10 plain rounds, 10 16-row rounds, {d} prefill chunks), bad words {d}, {d} ms\n", .{ calls, chunks, bad, (nowNs() - t0) / std.time.ns_per_ms });
    std.debug.print("{s}", .{line});
    try out.appendSlice(gpa, line);
    if (corrupt == null) {
        try time(&e, gpa, "dspark", &decode_round, rounds, 20, &out);
        try time(&e, gpa, "plain", &plain_round, rounds, 20, &out);
        try time(&e, gpa, "wide16", &wide_round, rounds, 20, &out);
        try time(&e, gpa, "prefill2048", &prefill_chunk, chunks, 1, &out);
    }
    if (graph_reps > 0) bad += try graphTest(&e, gpa, graph_reps, &out);
    if (corrupt) |cr| {
        const graphed = graph_reps > 0 and c.capturable;
        const want: usize = if (graphed) 2 else 1;
        const graph_call = (cr.call - 1) % passCalls(&decode_round) + 1;
        var ok = nhits == want and bad == want;
        for (hits[0..@min(nhits, hits.len)]) |h| {
            ok = ok and h.half == cr.rank and h.words == 1 and h.call == (if (h.graph) graph_call else cr.call);
            const hl = try std.fmt.allocPrint(gpa, "negative-control hit: {s} call {d}, half {d}, {d} bad words\n", .{ if (h.graph) "graph" else "validation", h.call, h.half, h.words });
            std.debug.print("{s}", .{hl});
            try out.appendSlice(gpa, hl);
        }
        const verdict = try std.fmt.allocPrint(gpa, "negative-control: rank={d} injected_by={d} call={d}{s} bad_words={d} places={d} want={d} {s}\n", .{ rank, cr.rank, cr.call, if (graphed) " (+graph)" else "", bad, nhits, want, if (ok) "PASS" else "FAIL" });
        std.debug.print("{s}", .{verdict});
        try out.appendSlice(gpa, verdict);
        if (receipt) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
        return if (ok) 0 else 4;
    }
    if (receipt) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
    return if (bad == 0) 0 else 1;
}
