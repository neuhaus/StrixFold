//! Eager HIP TP2 gather: stage, exchange and upload. See tools/dsv41-hip/COLLECTIVES-DESIGN.md.
const std = @import("std");
const gpu = @import("gpu");
const coll = @import("coll.zig");
const Fan = @import("fan.zig").Fan;
const Link = @import("link.zig").Link;
const wire = @import("tcp_wire.zig");
const HostWire = @import("coll_wire.zig").HostWire;

const c = std.c;
const hello_magic: u32 = 0x31484354; // "TCH1"

/// Settings both ranks must agree on (part of the handshake text).
pub const Staging = enum { blit, dma };

pub const Settings = struct {
    data_port: u16,
    staging: Staging = .dma,
    pipe_min: usize = 8 << 20,
    chunk: usize = 4 << 20,
    stripe: ?[4]u8,
    opt: wire.Options,
    /// a GPU wait longer than this fails: tcp_host's (this rank's block reaching pinned memory, the previous upload
    /// finishing) and tcp_proxy's device waits (TF_COLL_GPU_WAIT_S)
    gpu_wait_ns: u64 = 10 * std.time.ns_per_s,
    /// a GPU wait longer than this is logged when it ends (TF_COLL_GPU_WARN_S; 0: never); not in the handshake
    gpu_warn_ns: u64 = 2 * std.time.ns_per_s,
};

/// One GPU wait's clock, GPU-free (the host tests drive it): past `limit_ns` it has expired; `slow` once it is past
/// `warn_ns` (0: never).
pub const WaitClock = struct {
    t0: u64,
    limit_ns: u64,
    warn_ns: u64,

    pub fn expired(w: WaitClock, now: u64) bool {
        return now -| w.t0 > w.limit_ns;
    }

    pub fn slow(w: WaitClock, now: u64) bool {
        return w.warn_ns > 0 and now -| w.t0 > w.warn_ns;
    }

    pub fn ms(w: WaitClock, now: u64) u64 {
        return (now -| w.t0) / std.time.ns_per_ms;
    }
};

pub const TcpColl = struct {
    gpa: std.mem.Allocator,
    d: *const gpu.Driver,
    rank: u32,
    w: HostWire,
    tx: ?gpu.HostBuffer = null,
    rx: ?gpu.HostBuffer = null,
    staged: gpu.Event,
    uploaded: gpu.Event,
    /// pipelined gathers: their copies run on this stream, ordered after the caller's stream by `start`
    copy: gpu.Stream,
    start: gpu.Event,
    chunk_ready: [max_chunks]gpu.Event,
    pipe_min: usize,
    chunk: usize,
    upload_pending: bool = false,
    gpu_wait_ns: u64,
    gpu_warn_ns: u64,
    staging: Staging,
    /// device addresses of tx and rx (blit staging: mapped pinned memory)
    tx_dev: u64 = 0,
    rx_dev: u64 = 0,
    gpu_failed: std.atomic.Value(u32) = .init(0),
    reason: std.atomic.Value(u32) = .init(0),
    /// the gather begun by gatherBegin, which gatherEnd exchanges
    begun: ?Begun = null,
    /// the upload `uploaded` marks (its gather), for waitUploaded's log line
    uploading: Wait = .{ .op = "upload", .site = .{ .phase = .test_only, .layer = 0 }, .bytes = 0 },

    const vtable: coll.Coll.VTable = .{ .all_gather = allGather, .abort = abort, .close = close, .gather_begin = gatherBegin, .gather_end = gatherEnd, .check = check, .counts = counts };

    fn counts(ptr: *anyopaque) coll.Coll.Counts {
        const t: *TcpColl = @ptrCast(@alignCast(ptr));
        return .{ .host = t.w.seq() };
    }

    fn check(ptr: *anyopaque) coll.Error!void {
        const t: *TcpColl = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
    }
    const max_chunks = 64;

    /// A begun gather: its chunks' copies to pinned memory (and the own block's copy) queued on the copy stream.
    /// `whole`: blit staging, gathered in one piece at the end instead.
    const Begun = struct { site: coll.Site, send: u64, recv: u64, bytes: usize, chunk: usize, n: usize, whole: bool = false };

    pub fn coll_(t: *TcpColl) coll.Coll {
        return .{ .ptr = t, .vtable = &vtable, .rank = t.rank, .world = 2, .capturable = false };
    }

    fn ensure(t: *TcpColl, bytes: usize) coll.Error!void {
        if (t.tx) |b| if (b.bytes.len >= bytes) return;
        // grow both staging buffers (rare: the largest call sets the size); an upload still reading rx must finish first
        try t.waitUploaded();
        const n = std.mem.alignForward(usize, @max(bytes, 1 << 20), 1 << 20);
        if (t.tx) |*b| b.free();
        if (t.rx) |*b| b.free();
        t.tx = null;
        t.rx = null;
        const flags: gpu.HostFlags = if (t.staging == .blit) .{ .mapped = true, .coherent = true } else .{};
        t.tx = gpu.HostBuffer.allocFlags(t.d, n, flags) catch return t.gpuFail();
        t.rx = gpu.HostBuffer.allocFlags(t.d, n, flags) catch return t.gpuFail();
        if (t.staging == .blit) {
            t.tx_dev = t.tx.?.device() catch return t.gpuFail();
            t.rx_dev = t.rx.?.device() catch return t.gpuFail();
        }
    }

    fn gpuFail(t: *TcpColl) coll.Error {
        t.gpu_failed.store(1, .release);
        t.w.abort();
        return error.CollectiveFailed;
    }

    fn mapWire(t: *TcpColl, e: wire.Error) coll.Error {
        if (t.gpu_failed.load(.acquire) != 0) return error.CollectiveFailed;
        return switch (e) {
            error.PeerLost => error.PeerLost,
            error.Timeout => error.Timeout,
            error.Mismatch => error.Mismatch,
            error.Aborted => error.Aborted,
            error.SocketFailed => error.CollectiveFailed,
        };
    }

    fn status(t: *TcpColl) ?coll.Error {
        if (t.gpu_failed.load(.acquire) != 0) return error.CollectiveFailed;
        return if (t.w.status()) |e| t.mapWire(e) else null;
    }

    /// What a GPU wait is for, for its log line: this rank's block reaching pinned memory (staged), a pipelined
    /// gather's chunk k + 1 of n reaching it (chunk), or the previous gather's upload out of rx finishing (upload).
    const Wait = struct { op: []const u8, site: coll.Site, bytes: usize, chunk: usize = 0, chunks: usize = 1 };

    /// Spins on `ev` (no sleeps: these waits sit on the decode critical path) until it completes, the wire fails
    /// (abort, or the peer's failure seen earlier), or the GPU wait deadline passes. A wait that ends past the warn
    /// threshold, fails or passes the deadline logs one line as it ends.
    fn spinEvent(t: *TcpColl, ev: gpu.Event, what: Wait) coll.Error!void {
        const clock: WaitClock = .{ .t0 = wire.monotonicNs(), .limit_ns = t.gpu_wait_ns, .warn_ns = t.gpu_warn_ns };
        var spins: u32 = 0;
        while (true) {
            const ok = ev.done() catch return t.gpuFail();
            if (ok) {
                if (spins >= 256) {
                    const now = wire.monotonicNs();
                    if (clock.slow(now)) t.logWait(what, clock.ms(now), "done");
                }
                return;
            }
            spins +%= 1;
            if (spins % 256 == 0) {
                const now = wire.monotonicNs();
                if (t.status()) |e| {
                    t.logWait(what, clock.ms(now), @errorName(e));
                    return e;
                }
                if (clock.expired(now)) {
                    t.logWait(what, clock.ms(now), "past TF_COLL_GPU_WAIT_S, aborting the collective (Timeout)");
                    t.w.abort();
                    return error.Timeout;
                }
            }
            std.atomic.spinLoopHint();
        }
    }

    fn logWait(t: *TcpColl, what: Wait, ms: u64, outcome: []const u8) void {
        std.log.warn("collectives: rank {d} tcp_host waited {d} ms for its GPU ({s}, {s} layer {d}, {d} bytes, chunk {d} of {d}, after exchange {d}): {s}", .{
            t.rank, ms, what.op, @tagName(what.site.phase), what.site.layer, what.bytes, what.chunk + 1, what.chunks, t.w.seq(), outcome,
        });
    }

    /// One exchange on the wire; a failure logs one line (site, bytes, chunk, how long it ran, the wire's reason).
    fn exchange(t: *TcpColl, site: coll.Site, tx: []const u8, rx: []u8, chunk: usize, chunks: usize) coll.Error!void {
        const t0 = wire.monotonicNs();
        const site_word = (@as(u32, @intFromEnum(site.phase)) << 16) | site.layer;
        t.w.exchange(site_word, tx, rx) catch |e| {
            std.log.warn("collectives: rank {d} tcp_host exchange {d} failed after {d} ms ({s} layer {d}, {d} bytes, chunk {d} of {d}): {s} ({s})", .{
                t.rank, t.w.seq(), (wire.monotonicNs() -| t0) / std.time.ns_per_ms, @tagName(site.phase), site.layer, tx.len, chunk + 1, chunks, @errorName(e), t.w.detail(),
            });
            return t.mapWire(e);
        };
    }

    fn waitUploaded(t: *TcpColl) coll.Error!void {
        if (!t.upload_pending) return;
        try t.spinEvent(t.uploaded, t.uploading);
        t.upload_pending = false;
    }

    /// A call between gatherBegin and gatherEnd, or an end without a begin: the ranks' sequences can no longer be trusted,
    /// so the collective fails here (and the peer's next call with it).
    fn misuse(t: *TcpColl, what: []const u8) coll.Error {
        std.log.err("collectives: {s}", .{what});
        return t.gpuFail();
    }

    fn allGather(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *TcpColl = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        if (t.begun != null) return t.misuse("an all-gather while a begun gather is pending");
        if (bytes == 0) return;
        if (t.pipe_min > 0 and bytes >= t.pipe_min and t.staging == .dma) {
            try gatherBegin(ptr, stream, site, send, recv, bytes);
            return gatherEnd(ptr, stream);
        }
        try t.ensure(bytes);
        try t.waitUploaded(); // rx is about to be overwritten
        const tx = t.tx.?.bytes[0..bytes];
        const rx = t.rx.?.bytes[0..bytes];
        // this rank's block, once the stream reaches it
        if (t.staging == .blit) {
            gpu.copyD2DAsync(t.d, stream.handle, t.tx_dev, send, bytes) catch return t.gpuFail();
        } else {
            t.d.check(t.d.api.hipMemcpyDtoHAsync(@ptrCast(tx.ptr), gpu.abi.ptr(send), bytes, stream.handle), "hipMemcpyDtoHAsync") catch return t.gpuFail();
        }
        t.staged.record(stream) catch return t.gpuFail();
        try t.spinEvent(t.staged, .{ .op = "staged", .site = site, .bytes = bytes });
        try t.exchange(site, tx, rx, 0, 1);
        // the peer's block and this rank's own, in the stream's order
        const peer: u64 = 1 - t.rank;
        if (t.staging == .blit) {
            gpu.copyD2DAsync(t.d, stream.handle, recv + peer * bytes, t.rx_dev, bytes) catch return t.gpuFail();
        } else {
            gpu.uploadPinned(t.d, stream, recv + peer * bytes, rx) catch return t.gpuFail();
        }
        t.uploaded.record(stream) catch return t.gpuFail();
        t.upload_pending = true;
        t.uploading = .{ .op = "upload", .site = site, .bytes = bytes };
        const own = recv + @as(u64, t.rank) * bytes;
        if (own != send) gpu.copyD2DAsync(t.d, stream.handle, own, send, bytes) catch return t.gpuFail();
    }

    /// The first half of a gather (coll.Coll.gatherBegin), and of every large one (>= pipe_min, in chunks): after the
    /// caller's stream reaches this point, the copy stream copies each chunk to pinned memory and then the own block
    /// into `recv`; nothing waits here. Both ranks split the same size the same way (the chunk size is in the
    /// handshake), so their sequences stay in step.
    fn gatherBegin(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *TcpColl = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        if (t.begun != null) return t.misuse("a gather begun while another is pending");
        if (t.staging != .dma or bytes == 0) {
            t.begun = .{ .site = site, .send = send, .recv = recv, .bytes = bytes, .chunk = 0, .n = 0, .whole = true };
            return;
        }
        try t.ensure(bytes);
        const chunk = if (t.pipe_min > 0 and bytes >= t.pipe_min) @max(t.chunk, (bytes + max_chunks - 1) / max_chunks) else bytes;
        const n = (bytes + chunk - 1) / chunk;
        const tx = t.tx.?.bytes[0..bytes];
        t.start.record(stream) catch return t.gpuFail();
        t.copy.wait(t.start) catch return t.gpuFail();
        for (0..n) |k| {
            const off = k * chunk;
            const len = @min(chunk, bytes - off);
            t.d.check(t.d.api.hipMemcpyDtoHAsync(@ptrCast(tx.ptr + off), gpu.abi.ptr(send + off), len, t.copy.handle), "hipMemcpyDtoHAsync") catch return t.gpuFail();
            t.chunk_ready[k].record(t.copy) catch return t.gpuFail();
        }
        // the own block off the caller's stream, while the wire runs
        const own = recv + @as(u64, t.rank) * bytes;
        if (own != send) gpu.copyD2DAsync(t.d, t.copy.handle, own, send, bytes) catch return t.gpuFail();
        t.begun = .{ .site = site, .send = send, .recv = recv, .bytes = bytes, .chunk = chunk, .n = n };
    }

    /// The second half: each chunk is exchanged as soon as its copy lands and its upload is queued on the copy stream
    /// right behind, so copies overlap the wire; the caller's stream then waits for the last upload (and the own
    /// block's copy before it).
    fn gatherEnd(ptr: *anyopaque, stream: gpu.Stream) coll.Error!void {
        const t: *TcpColl = @ptrCast(@alignCast(ptr));
        const b = t.begun orelse return t.misuse("a gather ended that was never begun");
        t.begun = null;
        if (t.status()) |e| return e;
        if (b.whole) return allGather(ptr, stream, b.site, b.send, b.recv, b.bytes);
        try t.waitUploaded(); // rx is about to be overwritten
        const tx = t.tx.?.bytes[0..b.bytes];
        const rx = t.rx.?.bytes[0..b.bytes];
        const peer: u64 = 1 - t.rank;
        for (0..b.n) |k| {
            const off = k * b.chunk;
            const len = @min(b.chunk, b.bytes - off);
            try t.spinEvent(t.chunk_ready[k], .{ .op = "chunk", .site = b.site, .bytes = b.bytes, .chunk = k, .chunks = b.n });
            try t.exchange(b.site, tx[off..][0..len], rx[off..][0..len], k, b.n);
            gpu.uploadPinned(t.d, t.copy, b.recv + peer * b.bytes + off, rx[off..][0..len]) catch return t.gpuFail();
        }
        t.uploaded.record(t.copy) catch return t.gpuFail();
        t.upload_pending = true;
        t.uploading = .{ .op = "upload", .site = b.site, .bytes = b.bytes, .chunk = b.n - 1, .chunks = b.n };
        stream.wait(t.uploaded) catch return t.gpuFail();
    }

    fn abort(ptr: *anyopaque, reason: coll.Reason) void {
        const t: *TcpColl = @ptrCast(@alignCast(ptr));
        _ = t.reason.cmpxchgStrong(0, @as(u32, @intFromEnum(reason)) + 1, .acq_rel, .acquire);
        t.w.abort();
    }

    fn close(ptr: *anyopaque) void {
        const t: *TcpColl = @ptrCast(@alignCast(ptr));
        t.w.close(t.gpa);
        // the copy stream may still write tx (a begun or failed chunked gather's copies) or read rx: a bounded wait
        // for it before the buffers go, else they leak below
        const copy_idle = if (t.copy.synchronizeFor(2 * std.time.ns_per_s)) |_| true else |_| false;
        // an upload may still read rx: give it a bounded time before the buffers go
        if (t.upload_pending) {
            const t0 = wire.monotonicNs();
            while (wire.monotonicNs() - t0 < 2 * std.time.ns_per_s) {
                const ok = t.uploaded.done() catch break;
                if (ok) {
                    t.upload_pending = false;
                    break;
                }
            }
        }
        if (copy_idle and !t.upload_pending) {
            if (t.tx) |*b| b.free();
            if (t.rx) |*b| b.free();
        } // else: leaked on purpose; freeing memory a DMA may still read is worse
        t.staged.deinit();
        t.uploaded.deinit();
        t.start.deinit();
        for (&t.chunk_ready) |*e| e.deinit();
        t.copy.deinit();
        const gpa = t.gpa;
        gpa.destroy(t);
    }
};

fn envInt(comptime T: type, name: [:0]const u8, default: T) !T {
    const v = c.getenv(name) orelse return default;
    return std.fmt.parseInt(T, std.mem.span(v), 10) catch error.BadSetting;
}

/// The settings from the environment, with the control link's own address and port as the defaults.
pub fn settingsFromEnv(control_port: u16) !Settings {
    var s: Settings = .{
        .data_port = try envInt(u16, "TF_TP_DATA_PORT", control_port +% 16),
        .stripe = null,
        .opt = .{},
    };
    if (c.getenv("TF_TP_STRIPE")) |v| s.stripe = try @import("link.zig").parseIp(std.mem.span(v));
    s.opt.stripe_min = (try envInt(usize, "TF_COLL_STRIPE_MIB", 4)) << 20;
    s.opt.rto_min_us = try envInt(c_int, "TF_COLL_RTO_MIN_US", 8000);
    s.opt.progress_ns = (try envInt(u64, "TF_COLL_PROGRESS_MS", 2000)) * std.time.ns_per_ms;
    if (c.getenv("TF_COLL_STAGING")) |v| s.staging = std.meta.stringToEnum(Staging, std.mem.span(v)) orelse return error.BadSetting;
    s.pipe_min = (try envInt(usize, "TF_COLL_PIPE_MIB", 8)) << 20;
    s.chunk = @max(1, try envInt(usize, "TF_COLL_CHUNK_MIB", 4)) << 20;
    s.gpu_wait_ns = try envSeconds("TF_COLL_GPU_WAIT_S", 10, 1, 600);
    s.gpu_warn_ns = try envSeconds("TF_COLL_GPU_WARN_S", 2, 0, 600);
    return s;
}

/// Whole seconds in [lo, hi] as nanoseconds (`text` null: the default); BadSetting otherwise.
pub fn parseSeconds(text: ?[]const u8, default: u64, lo: u64, hi: u64) !u64 {
    const v = if (text) |t| (std.fmt.parseInt(u64, t, 10) catch return error.BadSetting) else default;
    if (v < lo or v > hi) return error.BadSetting;
    return v * std.time.ns_per_s;
}

fn envSeconds(name: [:0]const u8, default: u64, lo: u64, hi: u64) !u64 {
    const v = c.getenv(name);
    return parseSeconds(if (v) |p| std.mem.span(p) else null, default, lo, hi) catch |e| {
        std.log.err("collectives: {s}={s}: whole seconds from {d} to {d}", .{ name, if (v) |p| std.mem.span(p) else "", lo, hi });
        return e;
    };
}

/// The control link's two ends: rank 0's address (the data links listen there) and the control port.
fn controlEnds(fd: c.fd_t, rank: u32) !struct { ip: [4]u8, port: u16 } {
    var a: c.sockaddr.in = undefined;
    var len: c.socklen_t = @sizeOf(c.sockaddr.in);
    const rc = if (rank == 0) c.getsockname(fd, @ptrCast(&a), &len) else c.getpeername(fd, @ptrCast(&a), &len);
    if (rc != 0 or a.family != c.AF.INET) return error.NoControlAddress;
    return .{ .ip = @bitCast(a.addr), .port = std.mem.bigToNative(u16, a.port) };
}

/// This rank's handshake text: the identity and the transport settings, as space-separated key=value pairs.
fn identityText(buf: []u8, id: *const coll.Identity, s: Settings, data_links: usize) ![]u8 {
    return std.fmt.bufPrint(buf, "world={d} kit_abi={d} kit_sha256={s} modalities={d} {s} wire={d} data_port={d} links={d} stripe_mib={d} rto_min_us={d} progress_ms={d} gpu_wait_s={d} staging={s} pipe_mib={d} chunk_mib={d}", .{
        id.world, id.kit_abi, &id.kit_sha256, id.modalities, id.plan, wire.version, s.data_port, data_links,
        s.opt.stripe_min >> 20, s.opt.rto_min_us, s.opt.progress_ns / std.time.ns_per_ms, s.gpu_wait_ns / std.time.ns_per_s, @tagName(s.staging), s.pipe_min >> 20, s.chunk >> 20,
    });
}

/// Logs every key whose value differs between the two texts (or that only one side has); true when they are equal.
fn compareText(mine: []const u8, theirs: []const u8) bool {
    var equal = true;
    var it = std.mem.tokenizeScalar(u8, mine, ' ');
    while (it.next()) |kv| {
        const key = kv[0 .. std.mem.indexOfScalar(u8, kv, '=') orelse kv.len];
        const other = findKey(theirs, key);
        if (other == null or !std.mem.eql(u8, other.?, kv)) {
            std.log.warn("collectives handshake: this rank has {s}, the peer has {s}", .{ kv, other orelse "(nothing)" });
            equal = false;
        }
    }
    var it2 = std.mem.tokenizeScalar(u8, theirs, ' ');
    while (it2.next()) |kv| {
        const key = kv[0 .. std.mem.indexOfScalar(u8, kv, '=') orelse kv.len];
        if (findKey(mine, key) == null) {
            std.log.warn("collectives handshake: the peer has {s}, this rank has nothing", .{kv});
            equal = false;
        }
    }
    return equal;
}

fn findKey(text: []const u8, key: []const u8) ?[]const u8 {
    var it = std.mem.tokenizeScalar(u8, text, ' ');
    while (it.next()) |kv| {
        if (kv.len > key.len and kv[key.len] == '=' and std.mem.eql(u8, kv[0..key.len], key)) return kv;
        if (kv.len == key.len and std.mem.eql(u8, kv, key)) return kv;
    }
    return null;
}

/// What the peers' handshake settles: the session epoch, and the data links rank 0 listens on.
pub const Session = struct {
    epoch: [16]u8,
    addrs: [wire.max_links][4]u8,
    links: usize,
    settings: Settings,
};

/// The peers' handshake on the control link: hello (magic, rank, a fresh nonce) both ways, then each rank's identity
/// text (coll.Identity, this transport's settings and `extra`), compared key by key. Both ranks send first (small
/// frames). Refuses with Mismatch, logging every key that differs.
pub fn handshake(fan: *const Fan, id: *const coll.Identity, extra: []const u8) !Session {
    if (id.world != 2) return error.UnsupportedWorld;
    const link: Link = fan.links[0] orelse return error.NoControlLink;
    const ends = try controlEnds(link.fd, id.rank);
    var se: Session = .{ .epoch = undefined, .addrs = undefined, .links = 1, .settings = try settingsFromEnv(ends.port) };
    se.addrs[0] = ends.ip;
    if (se.settings.stripe) |ip| {
        se.addrs[1] = ip;
        se.links = 2;
    }
    var nonce: [16]u8 = undefined;
    if (std.os.linux.getrandom(&nonce, nonce.len, 0) != nonce.len) return error.NoRandom;
    var hello: [24]u8 = undefined;
    std.mem.writeInt(u32, hello[0..4], hello_magic, .little);
    std.mem.writeInt(u32, hello[4..8], id.rank, .little);
    @memcpy(hello[8..24], &nonce);
    try link.send(&hello);
    var text_buf: [2048]u8 = undefined; // (the line is printed twice into it; the line itself stays under the peer's 1280)
    const base = try identityText(&text_buf, id, se.settings, se.links);
    const mine = try std.fmt.bufPrint(text_buf[base.len..], "{s} {s}", .{ base, extra });
    // bufPrint into the tail of the same buffer: copy the whole line to the front first
    std.mem.copyForwards(u8, text_buf[0..mine.len], mine);
    const line = text_buf[0..mine.len];
    try link.send(line);
    var in_buf: [1280]u8 = undefined;
    const peer_hello = try link.recv(&in_buf);
    if (peer_hello.len != 24 or std.mem.readInt(u32, peer_hello[0..4], .little) != hello_magic or
        std.mem.readInt(u32, peer_hello[4..8], .little) != 1 - id.rank) return error.Mismatch;
    for (&se.epoch, nonce, peer_hello[8..24]) |*e, a, b| e.* = a ^ b;
    if (std.mem.allEqual(u8, &se.epoch, 0)) return error.Mismatch;
    var peer_buf: [1280]u8 = undefined;
    const theirs = try link.recv(&peer_buf);
    if (!compareText(line, theirs)) {
        std.log.err("collectives handshake refused: the ranks differ in the settings listed above", .{});
        return error.Mismatch;
    }
    return se;
}

/// tcp_host over the session's data links (`port_offset` past the data port: the proxy keeps its own wire).
pub fn openHost(gpa: std.mem.Allocator, d: *const gpu.Driver, rank: u32, se: Session, port_offset: u16) !*TcpColl {
    var w: HostWire = .{ .tcp = try wire.Wire.open(@intCast(rank), se.addrs[0..se.links], se.settings.data_port + port_offset, se.epoch, se.settings.opt) };
    errdefer w.close(gpa);
    const s = se.settings;
    const t = try initHost(gpa, d, rank, s, w);
    std.log.info("collectives: tcp_host over {d} data link(s) from port {d}, stripe >= {d} MiB, RTO floor {d} us, progress deadline {d} ms, GPU waits fail after {d} s (logged past {d} s), {s} staging, gathers >= {d} MiB pipelined in {d} MiB chunks", .{ se.links, s.data_port + port_offset, s.opt.stripe_min >> 20, s.opt.rto_min_us, s.opt.progress_ns / std.time.ns_per_ms, s.gpu_wait_ns / std.time.ns_per_s, s.gpu_warn_ns / std.time.ns_per_s, @tagName(s.staging), s.pipe_min >> 20, s.chunk >> 20 });
    return t;
}

/// The eager/pipelined GPU staging path on its own QP. All payloads use verbs in this mode.
pub fn openVerbsHost(gpa: std.mem.Allocator, d: *const gpu.Driver, rank: u32, se: Session, link: Link, device: []const u8, gid_index: u8) !*TcpColl {
    const s = se.settings;
    const v = try @import("verbs_wire.zig").Wire.open(gpa, link, device, gid_index, @intCast(rank), se.epoch, s.chunk, s.opt.progress_ns);
    errdefer v.close(gpa);
    return initHost(gpa, d, rank, s, .{ .verbs = v });
}

fn initHost(gpa: std.mem.Allocator, d: *const gpu.Driver, rank: u32, s: Settings, w: HostWire) !*TcpColl {
    const t = try gpa.create(TcpColl);
    errdefer gpa.destroy(t);
    var staged = try gpu.Event.init(d, false);
    errdefer staged.deinit();
    var uploaded = try gpu.Event.init(d, false);
    errdefer uploaded.deinit();
    var start = try gpu.Event.init(d, false);
    errdefer start.deinit();
    var copy = try gpu.Stream.init(d, true);
    errdefer copy.deinit();
    var ready: [TcpColl.max_chunks]gpu.Event = undefined;
    var made: usize = 0;
    errdefer for (ready[0..made]) |*e| e.deinit();
    while (made < ready.len) : (made += 1) ready[made] = try gpu.Event.init(d, false);
    t.* = .{ .gpa = gpa, .d = d, .rank = rank, .w = w, .staged = staged, .uploaded = uploaded, .gpu_wait_ns = s.gpu_wait_ns, .gpu_warn_ns = s.gpu_warn_ns, .staging = s.staging, .copy = copy, .start = start, .chunk_ready = ready, .pipe_min = s.pipe_min, .chunk = s.chunk };
    return t;
}

/// HIP transport selection. The graph-capable proxies share the same staging and collect kernels.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const gpu.Context, kit: ?*const coll.Kit, fan: *const Fan, id: *const coll.Identity) anyerror!coll.Coll {
    const kind = if (c.getenv("TF_COLL_KIND")) |v| std.mem.span(v) else "host";
    const use_verbs = std.mem.eql(u8, kind, "verbs");
    if (use_verbs or std.mem.eql(u8, kind, "proxy")) {
        const k = kit orelse return error.ProxyNeedsKit;
        return @import("tcp_proxy.zig").open(gpa, io, ctx, k, fan, id, use_verbs);
    }
    if (!std.mem.eql(u8, kind, "host")) return error.BadSetting;
    const se = try handshake(fan, id, "kind=host");
    const t = try openHost(gpa, ctx.d, id.rank, se, 0);
    return t.coll_();
}

test "the handshake text names every differing setting" {
    try std.testing.expect(compareText("a=1 b=2", "b=2 a=1"));
    try std.testing.expect(!compareText("a=1 b=2", "a=1 b=3"));
    try std.testing.expect(!compareText("a=1", "a=1 c=4"));
    try std.testing.expectEqualStrings("b=2", findKey("a=1 b=2", "b").?);
    try std.testing.expect(findKey("ab=1", "a") == null);
}

test "TF_COLL_GPU_WAIT_S and TF_COLL_GPU_WARN_S: whole seconds in range, the defaults when unset" {
    const ns = std.time.ns_per_s;
    try std.testing.expectEqual(@as(u64, 10 * ns), try parseSeconds(null, 10, 1, 600));
    try std.testing.expectEqual(@as(u64, 60 * ns), try parseSeconds("60", 10, 1, 600));
    try std.testing.expectEqual(@as(u64, 1 * ns), try parseSeconds("1", 10, 1, 600));
    try std.testing.expectEqual(@as(u64, 600 * ns), try parseSeconds("600", 10, 1, 600));
    try std.testing.expectEqual(@as(u64, 0), try parseSeconds("0", 2, 0, 600)); // the warn line off
    for ([_][]const u8{ "0", "601", "-5", "1.5", "", "60s", " 60" }) |bad|
        try std.testing.expectError(error.BadSetting, parseSeconds(bad, 10, 1, 600));

    const env = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    defer _ = env.unsetenv("TF_COLL_GPU_WAIT_S");
    defer _ = env.unsetenv("TF_COLL_GPU_WARN_S");
    _ = env.unsetenv("TF_COLL_GPU_WAIT_S");
    _ = env.unsetenv("TF_COLL_GPU_WARN_S");
    var s = try settingsFromEnv(29600);
    try std.testing.expectEqual(@as(u64, 10 * ns), s.gpu_wait_ns); // unchanged when unset
    try std.testing.expectEqual(@as(u64, 2 * ns), s.gpu_warn_ns);
    _ = env.setenv("TF_COLL_GPU_WAIT_S", "60", 1);
    _ = env.setenv("TF_COLL_GPU_WARN_S", "0", 1);
    s = try settingsFromEnv(29600);
    try std.testing.expectEqual(@as(u64, 60 * ns), s.gpu_wait_ns);
    try std.testing.expectEqual(@as(u64, 0), s.gpu_warn_ns);
}

test "a GPU wait expires only past its deadline and is slow once past the warn threshold" {
    const ms = std.time.ns_per_ms;
    const w: WaitClock = .{ .t0 = 1000 * ms, .limit_ns = 10_000 * ms, .warn_ns = 2000 * ms };
    try std.testing.expect(!w.expired(1000 * ms) and !w.slow(1000 * ms));
    try std.testing.expect(!w.slow(3000 * ms) and w.slow(3000 * ms + 1)); // strictly past 2 s
    try std.testing.expect(!w.expired(11_000 * ms) and w.expired(11_000 * ms + 1)); // strictly past 10 s
    try std.testing.expectEqual(@as(u64, 10_000), w.ms(11_000 * ms + 1));
    try std.testing.expect(!w.expired(0) and w.ms(0) == 0); // a clock read before t0 counts as no time
    const quiet: WaitClock = .{ .t0 = 0, .limit_ns = 60 * std.time.ns_per_s, .warn_ns = 0 };
    try std.testing.expect(!quiet.slow(59 * std.time.ns_per_s) and quiet.expired(61 * std.time.ns_per_s));
}

test "the GPU wait deadline is in the handshake, the warn threshold is not" {
    const id: coll.Identity = .{ .rank = 0, .world = 2, .kit_abi = 1, .kit_sha256 = @splat('0'), .modalities = 0, .plan = "p=1" };
    const base: Settings = .{ .data_port = 29616, .stripe = null, .opt = .{} };
    var b0: [1280]u8 = undefined;
    var b1: [1280]u8 = undefined;
    const ten = try identityText(&b0, &id, base, 1);
    try std.testing.expect(std.mem.indexOf(u8, ten, " gpu_wait_s=10 ") != null);
    var sixty = base;
    sixty.gpu_wait_ns = 60 * std.time.ns_per_s;
    try std.testing.expect(!compareText(ten, try identityText(&b1, &id, sixty, 1)));
    var loud = base;
    loud.gpu_warn_ns = 0;
    try std.testing.expect(compareText(ten, try identityText(&b1, &id, loud, 1)));
}
