//! tcp_proxy: the TP2 all-gather as two kernels on the model stream and a host proxy thread
//! (tools/dsv41-hip/COLLECTIVES-DESIGN.md sections 2.5 and 3.3), so a gather needs no host call on the stream's critical
//! path and can be captured into a HIP graph (capturable = true).
//!
//! Per gather: coll.stage copies this rank's block into a transmit slot of a mapped coherent slab and into its own
//! place in recv, then publishes the slot (ready = seq, with the size and site); the proxy thread, spinning on the
//! slot, exchanges it with the peer over its own tcp_wire link (thunderbolt0) into the slot's receive half and sets
//! done = seq; coll.collect waits for done (one guard load per block), copies the peer's block into recv and releases
//! the slot (consumed = seq). The sequence number lives in the slab and the stage kernel advances it, so graph replays
//! run fresh gathers. Gathers larger than a slot (prefill) take the tcp_host path over separate data links; they are
//! eager by nature, and one inside a capture is refused.
//!
//! Failure: the proxy poisons the collective on any wire error and sets the slab's abort word, which ends every device
//! wait at once; a device wait past its deadline (TF_COLL_GPU_WAIT_S) sets the abort word itself and the proxy then
//! fails the collective. The host raises the word as 1 and the device as 2, neither over the other, so the proxy can tell (and
//! log, one line) which side gave up; a failed exchange, and one past TF_COLL_GPU_WARN_S, logs a line too.
//! Settings (environment, both ranks, in the handshake): TF_COLL_SLOTS (8), TF_COLL_SLOT_MIB (4.25: a 16-row head
//! gather fits), TF_COLL_PROXY_CPU (6), TF_COLL_PROXY_STRIPE_KIB (0: every slot gather on the first cable; N: with
//! TF_TP_STRIPE, a slot gather of at least N KiB is split, its second half moved over the second cable by a second
//! proxy thread on TF_COLL_PROXY_CPU2, default 7), plus tcp_coll.zig's.
const std = @import("std");
const gpu = @import("gpu");
const coll = @import("coll.zig");
const Fan = @import("fan.zig").Fan;
const wire = @import("tcp_wire.zig");
const verbs = @import("verbs_wire.zig");
const tcp_coll = @import("tcp_coll.zig");

const c = std.c;
pub const max_slots = 16;

/// abi.h's tf_ds_coll_slot, field for field (checked below).
pub const Slot = extern struct { ready: u64, done: u64, consumed: u64, bytes: u64, site: u64, pad: [3]u64 };
/// abi.h's tf_ds_coll_shared, field for field (checked below).
pub const Shared = extern struct {
    abort: u32,
    pad0: u32,
    seq: u64,
    staged: u32,
    collected: u32,
    pad1: [5]u64,
    slot: [max_slots]Slot,
};

/// The slab header has one definition for hipcc and Zig: abi.h's structs. Every field must sit where abi.h puts it.
fn sameLayout(comptime Z: type, comptime C: type) void {
    if (@sizeOf(Z) != @sizeOf(C)) @compileError(@typeName(Z) ++ " and abi.h's " ++ @typeName(C) ++ " differ in size");
    for (@typeInfo(Z).@"struct".field_names) |name| {
        if (@offsetOf(Z, name) != @offsetOf(C, name)) @compileError(@typeName(Z) ++ "." ++ name ++ " is not where abi.h puts it");
    }
}

comptime {
    const C = gpu.kernel_abi;
    sameLayout(Slot, C.tf_ds_coll_slot);
    sameLayout(Shared, C.tf_ds_coll_shared);
    if (C.TF_DS_COLL_MAX_SLOTS != max_slots) @compileError("abi.h TF_DS_COLL_MAX_SLOTS is not tcp_proxy.max_slots");
}

/// The kernels' parameter lists, in coll.hip's order.
pub const params = struct {
    pub const Stage = extern struct { s: u64, send: u64, tx: u64, own: u64, bytes: u64, slot_bytes: u64, nslot: u32, site: u32, deadline: u64 };
    pub const Collect = extern struct { s: u64, rx: u64, peer: u64, bytes: u64, slot_bytes: u64, nslot: u32, deadline: u64 };
};

const StageK = gpu.Kernel("coll.stage", params.Stage);
const CollectK = gpu.Kernel("coll.collect", params.Collect);

/// The proxy's loop on a slab, GPU-free (the host test drives it with threads standing in for the kernels): wait for
/// slot q's ready, exchange its transmit half with the peer into its receive half, publish done, next q. A gather of
/// at least `stripe_min` bytes, with a `second` thread, is split: the second half goes to that thread (the other
/// cable) while this one moves the first half, and done waits for both. Returns the failure code (0: stopped cleanly)
/// after setting the abort word on any failure. Each failure, an exchange past `warn_ns` and a device wait that gave up
/// log one line (rank, gather, site, bytes, ms).
const DataWire = union(enum) {
    tcp: *wire.Wire,
    verbs: *verbs.Wire,

    fn exchange(w: DataWire, site: u32, out: []const u8, in: []u8) wire.Error!void {
        return switch (w) {
            .tcp => |t| t.exchange(site, out, in),
            .verbs => |v| v.exchange(site, out, in),
        };
    }

    fn detail(w: DataWire) []const u8 {
        return switch (w) {
            .tcp => |t| if (t.status() != null) std.mem.sliceTo(&t.why, 0) else "",
            .verbs => "RC verbs",
        };
    }
};

pub const Core = struct {
    shared: *Shared,
    tx: []u8, // nslot * slot_bytes
    rx: []u8,
    nslot: u32,
    slot_bytes: u64,
    w: DataWire,
    stop: *std.atomic.Value(u32),
    exchanged: *std.atomic.Value(u64),
    second: ?*Second = null,
    stripe_min: u64 = std.math.maxInt(u64),
    /// for the log lines: this rank, and an exchange longer than this is logged (0: never)
    rank: u32 = 0,
    warn_ns: u64 = 0,

    pub fn run(k: Core) u32 {
        var q: u64 = 1;
        var idle_since: u64 = 0;
        while (k.stop.load(.acquire) == 0) {
            const slot: usize = @intCast(q % k.nslot);
            const r = @atomicLoad(u64, &k.shared.slot[slot].ready, .acquire);
            if (r != q) {
                if (r > q) { // a generation ahead of the one expected: stale or duplicated
                    std.log.warn("collectives: rank {d} proxy expected gather {d} in slot {d} and found gather {d}", .{ k.rank, q, slot, r });
                    return k.fail(3);
                }
                if (@atomicLoad(u32, &k.shared.abort, .acquire) != 0) {
                    deviceGaveUp(k.shared, k.rank, q, k.nslot, "waiting for its GPU to stage gather");
                    return k.fail(2);
                }
                if (idle_since == 0) idle_since = wire.monotonicNs();
                if (wire.monotonicNs() - idle_since > 20 * std.time.ns_per_ms) std.Thread.yield() catch {} else std.atomic.spinLoopHint();
                continue;
            }
            idle_since = 0;
            const bytes: usize = @intCast(k.shared.slot[slot].bytes);
            const site: u32 = @truncate(k.shared.slot[slot].site);
            if (bytes > k.slot_bytes) {
                std.log.warn("collectives: rank {d} proxy: gather {d} says {d} bytes, more than a slot's {d}", .{ k.rank, q, bytes, k.slot_bytes });
                return k.fail(3);
            }
            const base = slot * k.slot_bytes;
            const second = if (bytes >= k.stripe_min) k.second else null;
            const first = if (second != null) splitAt(bytes) else bytes;
            if (second) |s| s.hand(q, site, base + first, bytes - first);
            const t0 = wire.monotonicNs();
            k.w.exchange(site, k.tx[base..][0..first], k.rx[base..][0..first]) catch |e| {
                logExchange(k.w.detail(), k.rank, 1, q, site, first, t0, @errorName(e));
                return k.fail(wireCode(e));
            };
            if (second) |s| {
                while (s.done.load(.acquire) != q) {
                    if (@atomicLoad(u32, &k.shared.abort, .acquire) != 0) {
                        deviceGaveUp(k.shared, k.rank, q, k.nslot, "waiting for the second cable's half of gather");
                        return k.fail(2);
                    }
                    if (k.stop.load(.acquire) != 0) return 0;
                    std.atomic.spinLoopHint();
                }
            }
            if (k.warn_ns > 0 and wire.monotonicNs() -| t0 > k.warn_ns) logExchange(k.w.detail(), k.rank, 1, q, site, bytes, t0, "done (this rank's coll.collect waited for it)");
            @atomicStore(u64, &k.shared.slot[slot].done, q, .release);
            _ = k.exchanged.fetchAdd(1, .monotonic);
            q += 1;
        }
        return 0;
    }

    fn fail(k: Core, code: u32) u32 {
        raiseAbort(k.shared);
        return code;
    }
};

/// Sets the abort word to 1 unless it is set: a device wait's 2 (coll.hip tf_ds_coll_fail) stays for deviceGaveUp.
fn raiseAbort(shared: *Shared) void {
    _ = @cmpxchgStrong(u32, &shared.abort, 0, 1, .acq_rel, .acquire);
}

/// When a device wait set the abort word (2: past its deadline, TF_COLL_GPU_WAIT_S, or a slot's done past its own
/// gather), the first proxy thread to see it logs one line with the slot's words and marks it logged (3); 1 (the host
/// raised it) was logged where it was raised. coll.stage waits for consumed + nslot >= its gather, coll.collect for
/// done == seq.
fn deviceGaveUp(shared: *Shared, rank: u32, q: u64, nslot: u32, what: []const u8) void {
    if (@cmpxchgStrong(u32, &shared.abort, 2, 3, .acq_rel, .acquire) != null) return;
    const slot: usize = @intCast(q % nslot);
    const w = &shared.slot[slot];
    const last = @atomicLoad(u64, &shared.seq, .acquire);
    const lw = &shared.slot[@intCast(last % nslot)];
    const site: u32 = @truncate(lw.site);
    std.log.warn("collectives: rank {d}: a device wait gave up (past TF_COLL_GPU_WAIT_S, or a done past its gather) while the proxy was {s} {d}: last staged {d} ({s} layer {d}, {d} bytes); slot {d} ready {d} done {d} consumed {d}", .{
        rank, what, q, last, phaseName(site), site & 0xffff, lw.bytes,
        slot, @atomicLoad(u64, &w.ready, .acquire), @atomicLoad(u64, &w.done, .acquire), @atomicLoad(u64, &w.consumed, .acquire),
    });
}

/// One line about a proxy exchange: a failure (`outcome` the error) or one past the warn threshold.
fn logExchange(detail: []const u8, rank: u32, cable: u8, q: u64, site: u32, bytes: usize, t0: u64, outcome: []const u8) void {
    std.log.warn("collectives: rank {d} proxy cable {d} exchange of gather {d} ({s} layer {d}, {d} bytes) after {d} ms: {s}{s}{s}", .{
        rank, cable, q, phaseName(site), site & 0xffff, bytes, (wire.monotonicNs() -| t0) / std.time.ns_per_ms, outcome,
        if (detail.len > 0) " " else "", detail,
    });
}

fn phaseName(site: u32) []const u8 {
    return if (std.enums.fromInt(coll.Phase, site >> 16)) |p| @tagName(p) else "phase ?";
}

/// Where a split gather's first half ends: half its bytes, rounded up to 64.
fn splitAt(bytes: usize) usize {
    return @min(bytes, std.mem.alignForward(usize, bytes / 2, 64));
}

fn wireCode(e: wire.Error) u32 {
    return switch (e) {
        error.PeerLost => 1,
        error.Timeout => 2,
        error.Mismatch => 3,
        error.Aborted => 4,
        error.SocketFailed => 5,
    };
}

/// The second proxy thread (TF_COLL_PROXY_STRIPE_KIB): it moves the second half of each split gather over the second
/// cable's own wire while the first thread moves the first half, so the two cables' TCP work runs on two cores (one
/// thread moves ~1.6 GB/s each way whatever the cables). The first thread hands a gather over in `todo` and waits for
/// `done`; this thread never reads the slab's slot words.
pub const Second = struct {
    w: *wire.Wire,
    tx: []u8,
    rx: []u8,
    shared: *Shared,
    stop: *std.atomic.Value(u32),
    /// the gather handed over (0: none yet); off, len and site are written before its release store
    todo: std.atomic.Value(u64) = .init(0),
    off: usize = 0,
    len: usize = 0,
    site: u32 = 0,
    /// the last gather whose second half has crossed both ways
    done: std.atomic.Value(u64) = .init(0),
    /// for the log lines (Core's)
    rank: u32 = 0,
    warn_ns: u64 = 0,
    nslot: u32 = 1,

    fn hand(s: *Second, q: u64, site: u32, off: usize, len: usize) void {
        s.off = off;
        s.len = len;
        s.site = site;
        s.todo.store(q, .release);
    }

    pub fn run(s: *Second) u32 {
        var last: u64 = 0;
        var idle_since: u64 = 0;
        while (s.stop.load(.acquire) == 0) {
            const q = s.todo.load(.acquire);
            if (q == last) {
                if (@atomicLoad(u32, &s.shared.abort, .acquire) != 0) {
                    deviceGaveUp(s.shared, s.rank, last + 1, s.nslot, "(second cable) idle before gather");
                    return s.fail(2);
                }
                if (idle_since == 0) idle_since = wire.monotonicNs();
                if (wire.monotonicNs() - idle_since > 20 * std.time.ns_per_ms) std.Thread.yield() catch {} else std.atomic.spinLoopHint();
                continue;
            }
            idle_since = 0;
            const t0 = wire.monotonicNs();
            s.w.exchange(s.site, s.tx[s.off..][0..s.len], s.rx[s.off..][0..s.len]) catch |e| {
                logExchange((DataWire{ .tcp = s.w }).detail(), s.rank, 2, q, s.site, s.len, t0, @errorName(e));
                return s.fail(wireCode(e));
            };
            if (s.warn_ns > 0 and wire.monotonicNs() -| t0 > s.warn_ns) logExchange((DataWire{ .tcp = s.w }).detail(), s.rank, 2, q, s.site, s.len, t0, "done");
            s.done.store(q, .release);
            last = q;
        }
        return 0;
    }

    fn fail(s: *Second, code: u32) u32 {
        raiseAbort(s.shared);
        return code;
    }
};

pub const TcpProxy = struct {
    gpa: std.mem.Allocator,
    d: *const gpu.Driver,
    rank: u32,
    host: *tcp_coll.TcpColl,
    w: ?wire.Wire,
    rdma: ?*verbs.Wire,
    control: ?*verbs.Wire,
    slab: gpu.HostBuffer,
    shared: *Shared,
    dev: u64, // the slab's device address
    tx_off: u64,
    rx_off: u64,
    nslot: u32,
    slot_bytes: u64,
    deadline_ticks: u64,
    stage: StageK,
    collect: CollectK,
    proxy_cpu: i32,
    thread: ?std.Thread = null,
    /// split gathers (TF_COLL_PROXY_STRIPE_KIB > 0 with a second cable): the second cable's wire and thread
    w2: ?wire.Wire = null,
    second: Second = undefined,
    stripe_min: u64 = std.math.maxInt(u64),
    proxy_cpu2: i32 = -1,
    thread2: ?std.Thread = null,
    stop: std.atomic.Value(u32) = .init(0),
    /// the proxy's failure (wire.Error as State + 1), 0 while healthy
    failed: std.atomic.Value(u32) = .init(0),
    /// gathers the proxy has completed (diagnostics)
    exchanged: std.atomic.Value(u64) = .init(0),
    /// the gather begun by gatherBegin: staged into a slot (collected at the end), or tcp_host's when it is larger
    begun: ?Begun = null,

    const Begun = struct { host: bool, recv: u64, bytes: usize };
    const vtable: coll.Coll.VTable = .{ .all_gather = allGather, .abort = abort, .close = close, .gather_begin = gatherBegin, .gather_end = gatherEnd, .check = check, .counts = counts };

    /// (read after the stream synchronized: the proxy threads' wire counts are then settled)
    fn counts(ptr: *anyopaque) coll.Coll.Counts {
        const t: *TcpProxy = @ptrCast(@alignCast(ptr));
        return .{ .staged = @atomicLoad(u64, &t.shared.seq, .acquire), .cable1 = t.exchanged.load(.acquire), .cable2 = if (t.w2) |w2| w2.seq else 0, .host = t.host.coll_().counts().host };
    }

    fn check(ptr: *anyopaque) coll.Error!void {
        const t: *TcpProxy = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
    }

    fn coll_(t: *TcpProxy) coll.Coll {
        return .{
            .ptr = t, .vtable = &vtable, .rank = t.rank, .world = 2, .capturable = true,
            .control = if (t.control != null) .{ .ptr = t, .send = sendControl, .recv = recvControl } else null,
        };
    }

    fn sendControl(ptr: *anyopaque, bytes: []const u8) coll.Error!void {
        const t: *TcpProxy = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        t.control.?.sendFrame(bytes) catch |e| {
            t.poison(wireCode(e));
            return t.status().?;
        };
    }

    fn recvControl(ptr: *anyopaque, buf: []u8, spin_ns: u64) coll.Error![]const u8 {
        const t: *TcpProxy = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        return t.control.?.recvFrame(buf, spin_ns) catch |e| {
            t.poison(wireCode(e));
            return t.status().?;
        };
    }

    fn txSlot(t: *TcpProxy, slot: usize) []u8 {
        return t.slab.bytes[t.tx_off + slot * t.slot_bytes ..][0..t.slot_bytes];
    }
    fn rxSlot(t: *TcpProxy, slot: usize) []u8 {
        return t.slab.bytes[t.rx_off + slot * t.slot_bytes ..][0..t.slot_bytes];
    }

    fn status(t: *TcpProxy) ?coll.Error {
        const f = t.failed.load(.acquire);
        if (f != 0) return switch (f) {
            1 => error.PeerLost,
            2 => error.Timeout,
            3 => error.Mismatch,
            4 => error.Aborted,
            else => error.CollectiveFailed,
        };
        if (@atomicLoad(u32, &t.shared.abort, .acquire) != 0) return error.Timeout; // a device wait gave up
        return null;
    }

    fn poison(t: *TcpProxy, code: u32) void {
        _ = t.failed.cmpxchgStrong(0, code, .acq_rel, .acquire);
        raiseAbort(t.shared);
        if (t.w) |*w| w.abort();
        if (t.rdma) |w| w.abort();
        if (t.control) |w| w.abort();
        if (t.w2) |*w2| w2.abort();
        t.host.coll_().abort(.cancelled);
    }

    fn pin(cpu_: i32) void {
        if (cpu_ < 0) return;
        var set: std.os.linux.cpu_set_t = @splat(0);
        const cpu: usize = @intCast(cpu_);
        set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
        std.os.linux.sched_setaffinity(0, &set) catch {};
    }

    fn proxyLoop(t: *TcpProxy) void {
        pin(t.proxy_cpu);
        const core: Core = .{
            .shared = t.shared,
            .tx = t.slab.bytes[t.tx_off..][0 .. t.nslot * t.slot_bytes],
            .rx = t.slab.bytes[t.rx_off..][0 .. t.nslot * t.slot_bytes],
            .nslot = t.nslot,
            .slot_bytes = t.slot_bytes,
            .w = if (t.rdma) |w| .{ .verbs = w } else .{ .tcp = &t.w.? },
            .stop = &t.stop,
            .exchanged = &t.exchanged,
            .second = if (t.w2 != null) &t.second else null,
            .stripe_min = t.stripe_min,
            .rank = t.rank,
            .warn_ns = t.host.gpu_warn_ns,
        };
        const code = core.run();
        if (code != 0) t.poison(code);
    }

    fn secondLoop(t: *TcpProxy) void {
        pin(t.proxy_cpu2);
        const code = t.second.run();
        if (code != 0) t.poison(code);
    }

    fn blocks(bytes: usize) u32 {
        return @intCast(std.math.clamp((bytes + 4095) / 4096, 1, 64));
    }

    fn allGather(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *TcpProxy = @ptrCast(@alignCast(ptr));
        if (t.begun != null) return t.misuse("an all-gather while a begun gather is pending");
        try gatherBegin(ptr, stream, site, send, recv, bytes);
        return gatherEnd(ptr, stream);
    }

    fn misuse(t: *TcpProxy, what: []const u8) coll.Error {
        std.log.err("collectives: {s}", .{what});
        t.poison(5);
        return error.CollectiveFailed;
    }

    /// The stage kernel (this rank's block into its slot, the proxy sends it as soon as it lands) or, for a gather
    /// larger than a slot, tcp_host's begin.
    fn gatherBegin(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const t: *TcpProxy = @ptrCast(@alignCast(ptr));
        if (t.status()) |e| return e;
        if (t.begun != null) return t.misuse("a gather begun while another is pending");
        if (bytes > t.slot_bytes) {
            const capturing = stream.capturing() catch return error.CollectiveFailed;
            if (capturing) {
                std.log.err("collectives: a {d}-byte gather inside a graph exceeds the {d}-byte slots (TF_COLL_SLOT_MIB)", .{ bytes, t.slot_bytes });
                t.poison(5);
                return error.CollectiveFailed;
            }
            t.host.coll_().vtable.gather_begin.?(t.host, stream, site, send, recv, bytes) catch |e| return t.hostFailure(e);
            t.begun = .{ .host = true, .recv = recv, .bytes = bytes };
            return;
        }
        t.begun = .{ .host = false, .recv = recv, .bytes = bytes };
        if (bytes == 0) return;
        const site_word = (@as(u32, @intFromEnum(site.phase)) << 16) | site.layer;
        const cfg: gpu.Config = .{ .grid = .{ .x = blocks(bytes), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
        t.stage.launch(stream, cfg, .{ .s = t.dev, .send = send, .tx = t.dev + t.tx_off, .own = recv + @as(u64, t.rank) * bytes, .bytes = bytes, .slot_bytes = t.slot_bytes, .nslot = t.nslot, .site = site_word, .deadline = t.deadline_ticks }) catch {
            t.poison(5);
            return error.CollectiveFailed;
        };
    }

    /// The collect kernel (waits for the proxy, then the peer's block into `recv`), or tcp_host's end.
    fn gatherEnd(ptr: *anyopaque, stream: gpu.Stream) coll.Error!void {
        const t: *TcpProxy = @ptrCast(@alignCast(ptr));
        const b = t.begun orelse return t.misuse("a gather ended that was never begun");
        t.begun = null;
        if (b.host) {
            t.host.coll_().vtable.gather_end.?(t.host, stream) catch |e| return t.hostFailure(e);
            return;
        }
        if (t.status()) |e| return e;
        if (b.bytes == 0) return;
        const peer: u64 = 1 - t.rank;
        const cfg: gpu.Config = .{ .grid = .{ .x = blocks(b.bytes), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
        t.collect.launch(stream, cfg, .{ .s = t.dev, .rx = t.dev + t.rx_off, .peer = b.recv + peer * b.bytes, .bytes = b.bytes, .slot_bytes = t.slot_bytes, .nslot = t.nslot, .deadline = t.deadline_ticks }) catch {
            t.poison(5);
            return error.CollectiveFailed;
        };
    }

    fn hostFailure(t: *TcpProxy, e: coll.Error) coll.Error {
        t.poison(switch (e) {
            error.PeerLost => 1,
            error.Timeout => 2,
            error.Mismatch => 3,
            error.Aborted => 4,
            error.CollectiveFailed => 5,
        });
        return e;
    }

    fn abort(ptr: *anyopaque, reason: coll.Reason) void {
        _ = reason;
        const t: *TcpProxy = @ptrCast(@alignCast(ptr));
        t.poison(4);
    }

    fn close(ptr: *anyopaque) void {
        const t: *TcpProxy = @ptrCast(@alignCast(ptr));
        t.stop.store(1, .release);
        @atomicStore(u32, &t.shared.abort, 1, .release); // ends any device wait
        if (t.w) |*w| w.close(std.time.ns_per_s);
        if (t.rdma) |w| w.abort();
        if (t.control) |w| w.abort();
        if (t.w2) |*w2| w2.close(std.time.ns_per_s);
        if (t.thread) |th| th.join();
        if (t.thread2) |th| th.join();
        if (t.rdma) |w| w.close(t.gpa);
        if (t.control) |w| w.close(t.gpa);
        // kernels still queued see the abort word and return without touching the slab's data
        _ = t.d.api.hipDeviceSynchronize();
        t.slab.free();
        t.host.coll_().vtable.close(t.host);
        const gpa = t.gpa;
        gpa.destroy(t);
    }
};

/// The device waits' deadline in wall-clock ticks (u128: 600 s at 100 MHz is past u64 as a product of nanoseconds).
pub fn deadlineTicks(wall_clock_hz: u64, wait_ns: u64) u64 {
    return @intCast(@as(u128, wall_clock_hz) * wait_ns / std.time.ns_per_s);
}

fn envU(name: [:0]const u8, default: u64) !u64 {
    const v = c.getenv(name) orelse return default;
    return std.fmt.parseInt(u64, std.mem.span(v), 10) catch error.BadSetting;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const gpu.Context, kit: *const gpu.Kit, fan: *const Fan, id: *const coll.Identity, use_verbs: bool) anyerror!coll.Coll {
    _ = io;
    const d = ctx.d;
    const nslot: u32 = @intCast(try envU("TF_COLL_SLOTS", 8));
    if (nslot < 2 or nslot > max_slots) return error.BadSetting;
    const slot_kib = try envU("TF_COLL_SLOT_KIB", 4352); // 4.25 MiB: a 16-row head gather (4,136,960 B) fits
    if (slot_kib == 0 or slot_kib > std.math.maxInt(u32) / 1024 - 4) return error.BadSetting;
    const slot_bytes = std.mem.alignForward(u64, slot_kib << 10, 4096);
    const proxy_cpu: i32 = @intCast(try envU("TF_COLL_PROXY_CPU", 6));
    const proxy_cpu2: i32 = @intCast(try envU("TF_COLL_PROXY_CPU2", 7));
    const stripe_kib = try envU("TF_COLL_PROXY_STRIPE_KIB", 0);
    if (use_verbs and (stripe_kib != 0 or c.getenv("TF_TP_STRIPE") != null)) return error.VerbsProxyStripingUnsupported;
    const device = if (use_verbs) std.mem.span(c.getenv("TF_COLL_VERBS_DEVICE") orelse return error.MissingVerbsDevice) else "";
    if (use_verbs and (device.len == 0 or device.len >= 64)) return error.BadSetting;
    const gid_index: u8 = if (use_verbs) std.math.cast(u8, try envU("TF_COLL_VERBS_GID_INDEX", 1)) orelse return error.BadSetting else 0;
    var extra_buf: [160]u8 = undefined;
    const extra = if (use_verbs)
        try std.fmt.bufPrint(&extra_buf, "kind=verbs-v2 slots={d} slot_kib={d} control_bytes={d}", .{ nslot, slot_kib, id.control_bytes })
    else
        try std.fmt.bufPrint(&extra_buf, "kind=proxy slots={d} slot_kib={d} proxy_stripe_kib={d}", .{ nslot, slot_kib, stripe_kib });
    const se = try tcp_coll.handshake(fan, id, extra);
    const rdma = if (use_verbs) try verbs.Wire.open(gpa, fan.links[0].?, device, gid_index,
        @intCast(id.rank), se.epoch, slot_bytes, se.settings.opt.progress_ns) else null;
    errdefer if (rdma) |v| v.close(gpa);
    // The TCP proxy's own links are separate from the host links used for oversized gathers.
    var w: ?wire.Wire = if (use_verbs) null else
        try wire.Wire.open(@intCast(id.rank), se.addrs[0..1], se.settings.data_port + 8, se.epoch, se.settings.opt);
    errdefer if (w) |*x| x.close(std.time.ns_per_s);
    var w2: ?wire.Wire = if (stripe_kib > 0 and se.links > 1)
        try wire.Wire.open(@intCast(id.rank), se.addrs[1..2], se.settings.data_port + 9, se.epoch, se.settings.opt)
    else
        null;
    errdefer if (w2) |*x| x.close(std.time.ns_per_s);
    const host = if (use_verbs) try tcp_coll.openVerbsHost(gpa, d, id.rank, se, fan.links[0].?, device, gid_index) else
        try tcp_coll.openHost(gpa, d, id.rank, se, 0);
    errdefer host.coll_().vtable.close(host);
    const control = if (use_verbs and id.control_bytes != 0)
        try verbs.Wire.openControl(gpa, fan.links[0].?, device, gid_index, @intCast(id.rank), se.epoch, id.control_bytes, se.settings.opt.progress_ns)
    else
        null;
    errdefer if (control) |v| v.close(gpa);
    const stage = try StageK.bind(kit, &.{});
    const collect = try CollectK.bind(kit, &.{});
    const caps = try ctx.caps();
    const tx_off: u64 = 4096;
    const rx_off: u64 = tx_off + nslot * slot_bytes;
    var slab = try gpu.HostBuffer.allocFlags(d, rx_off + nslot * slot_bytes, .{ .mapped = true, .coherent = true });
    errdefer slab.free();
    @memset(slab.bytes[0..@sizeOf(Shared)], 0);
    const t = try gpa.create(TcpProxy);
    errdefer gpa.destroy(t);
    t.* = .{
        .gpa = gpa,
        .d = d,
        .rank = id.rank,
        .host = host,
        .w = w,
        .rdma = rdma,
        .control = control,
        .slab = slab,
        .shared = @ptrCast(@alignCast(slab.bytes.ptr)),
        .dev = try slab.device(),
        .tx_off = tx_off,
        .rx_off = rx_off,
        .nslot = nslot,
        .slot_bytes = slot_bytes,
        .deadline_ticks = deadlineTicks(caps.wall_clock_hz, se.settings.gpu_wait_ns),
        .stage = stage,
        .collect = collect,
        .proxy_cpu = proxy_cpu,
        .w2 = w2,
        .stripe_min = if (w2 != null) stripe_kib << 10 else std.math.maxInt(u64),
        .proxy_cpu2 = proxy_cpu2,
    };
    if (t.w2) |*x| {
        t.second = .{ .w = x, .tx = t.slab.bytes[tx_off..][0 .. nslot * slot_bytes], .rx = t.slab.bytes[rx_off..][0 .. nslot * slot_bytes], .shared = t.shared, .stop = &t.stop, .rank = id.rank, .warn_ns = se.settings.gpu_warn_ns, .nslot = nslot };
        t.thread2 = try std.Thread.spawn(.{}, TcpProxy.secondLoop, .{t});
    }
    errdefer if (t.thread2) |th| {
        t.stop.store(1, .release);
        th.join();
    };
    t.thread = try std.Thread.spawn(.{}, TcpProxy.proxyLoop, .{t});
    if (use_verbs) {
        std.log.info("collectives: verbs proxy with {d} slots of {d} KiB, CPU {d}; oversized gathers use a separate verbs QP", .{ nslot, slot_bytes >> 10, proxy_cpu });
    } else if (t.w2 != null) {
        std.log.info("collectives: tcp_proxy with {d} slots of {d} KiB (coherent slab {d} MiB), proxy threads on CPUs {d} and {d}, gathers of {d} KiB and more split over both cables; larger gathers through tcp_host", .{ nslot, slot_bytes >> 10, slab.bytes.len >> 20, proxy_cpu, proxy_cpu2, stripe_kib });
    } else {
        std.log.info("collectives: tcp_proxy with {d} slots of {d} KiB (coherent slab {d} MiB), proxy thread on CPU {d}; larger gathers through tcp_host", .{ nslot, slot_bytes >> 10, slab.bytes.len >> 20, proxy_cpu });
    }
    return t.coll_();
}

// ---------------------------------------------------------------------------------------------------------------- tests
// The slab protocol without a GPU: two ranks' Cores over a socketpair, and host threads doing what coll.stage and
// coll.collect do (copy in, publish ready; wait for done, copy out, publish consumed), with the slot-reuse rule.

const testing = std.testing;

const FakeGpu = struct {
    shared: *Shared,
    tx: []u8,
    rx: []u8,
    nslot: u32,
    slot_bytes: u64,
    rank: u32,
    gathers: usize,
    bad: usize = 0,
    /// stage only, never collect (the peer's slots then fill up and its stage must wait)
    hold_collect_from: usize = std.math.maxInt(usize),
    timed_out: bool = false,

    fn pattern(q: u64, rank: u32, i: usize) u8 {
        return @truncate((q *% 131) ^ (@as(u64, rank) << 7) ^ i);
    }

    fn run(f: *FakeGpu) void {
        for (1..f.gathers + 1) |qi| {
            const q: u64 = qi;
            const slot: usize = @intCast(q % f.nslot);
            const bytes: usize = 1000 + (q % 7) * 4096;
            // stage: the slot is free once gather q - nslot was consumed
            const t0 = wire.monotonicNs();
            while (q > f.nslot and @atomicLoad(u64, &f.shared.slot[slot].consumed, .acquire) + f.nslot < q) {
                if (@atomicLoad(u32, &f.shared.abort, .acquire) != 0 or wire.monotonicNs() - t0 > 2 * std.time.ns_per_s) {
                    f.timed_out = true;
                    return;
                }
                std.atomic.spinLoopHint();
            }
            const tx = f.tx[slot * f.slot_bytes ..][0..bytes];
            for (tx, 0..) |*b, i| b.* = pattern(q, f.rank, i);
            f.shared.slot[slot].bytes = bytes;
            f.shared.slot[slot].site = 7;
            @atomicStore(u64, &f.shared.slot[slot].ready, q, .release);
            if (qi >= f.hold_collect_from) continue;
            // collect
            while (@atomicLoad(u64, &f.shared.slot[slot].done, .acquire) != q) {
                if (@atomicLoad(u32, &f.shared.abort, .acquire) != 0) return;
                std.atomic.spinLoopHint();
            }
            const rx = f.rx[slot * f.slot_bytes ..][0..bytes];
            for (rx, 0..) |b, i| f.bad += @intFromBool(b != pattern(q, 1 - f.rank, i));
            @atomicStore(u64, &f.shared.slot[slot].consumed, q, .release);
        }
    }
};

const Rank = struct {
    shared: Shared = std.mem.zeroes(Shared),
    tx: [4 * 32768]u8 = undefined,
    rx: [4 * 32768]u8 = undefined,
    w: wire.Wire,
    /// the second cable's wire and thread (split gathers), when the pair has one
    w2: ?wire.Wire = null,
    second: Second = undefined,
    stripe_min: u64 = std.math.maxInt(u64),
    stop: std.atomic.Value(u32) = .init(0),
    exchanged: std.atomic.Value(u64) = .init(0),
    code: u32 = 0,
    code2: u32 = 0,

    fn core(r: *Rank) Core {
        return .{ .shared = &r.shared, .tx = &r.tx, .rx = &r.rx, .nslot = 4, .slot_bytes = 32768, .w = .{ .tcp = &r.w }, .stop = &r.stop, .exchanged = &r.exchanged, .second = if (r.w2 != null) &r.second else null, .stripe_min = r.stripe_min };
    }

    fn runCore(r: *Rank) void {
        r.code = r.core().run();
    }

    fn runSecond(r: *Rank) void {
        r.code2 = r.second.run();
    }

    fn close(r: *Rank) void {
        r.w.close(std.time.ns_per_s);
        if (r.w2) |*w2| w2.close(std.time.ns_per_s);
    }
};

fn socketPair() ![2]c.fd_t {
    var sv: [2]c.fd_t = undefined;
    if (c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &sv) != 0) return error.SocketFailed;
    return sv;
}

/// The two ranks' wires over a socketpair; with `stripe_min`, a second socketpair as the second cable, and gathers of
/// at least that many bytes split over the two (Rank.runSecond moves the second halves).
fn rankPair(gpa: std.mem.Allocator, stripe_min: ?u64) ![2]*Rank {
    const epoch: [16]u8 = @splat(3);
    const opt: wire.Options = .{ .busy_poll_us = 0, .rto_min_us = 0, .progress_ns = 500 * std.time.ns_per_ms };
    const sv = try socketPair();
    const a = try gpa.create(Rank);
    const b = try gpa.create(Rank);
    a.* = .{ .w = wire.Wire.fromFds(0, &.{sv[0]}, epoch, opt) };
    b.* = .{ .w = wire.Wire.fromFds(1, &.{sv[1]}, epoch, opt) };
    if (stripe_min) |m| {
        const sv2 = try socketPair();
        a.w2 = wire.Wire.fromFds(0, &.{sv2[0]}, epoch, opt);
        b.w2 = wire.Wire.fromFds(1, &.{sv2[1]}, epoch, opt);
        for ([_]*Rank{ a, b }) |r| {
            r.stripe_min = m;
            r.second = .{ .w = &r.w2.?, .tx = &r.tx, .rx = &r.rx, .shared = &r.shared, .stop = &r.stop };
        }
    }
    return .{ a, b };
}

test "the slab protocol moves every gather through reused slots and stops cleanly" {
    try slabRun(null);
}

test "split gathers (TF_COLL_PROXY_STRIPE_KIB): larger ones in two halves over two wires and threads, smaller ones whole" {
    try slabRun(8192); // the fake gathers are 1,000 to 25,576 bytes: some split, some whole
}

fn slabRun(stripe_min: ?u64) !void {
    const gpa = testing.allocator;
    const rk = try rankPair(gpa, stripe_min);
    defer for (rk) |r| {
        r.close();
        gpa.destroy(r);
    };
    var g: [2]FakeGpu = undefined;
    for (&g, rk, 0..) |*f, r, i| f.* = .{ .shared = &r.shared, .tx = &r.tx, .rx = &r.rx, .nslot = 4, .slot_bytes = 32768, .rank = @intCast(i), .gathers = 2000 };
    var s: [2]?std.Thread = .{ null, null };
    if (stripe_min != null) for (&s, rk) |*th, r| {
        th.* = try std.Thread.spawn(.{}, Rank.runSecond, .{r});
    };
    const c0 = try std.Thread.spawn(.{}, Rank.runCore, .{rk[0]});
    const c1 = try std.Thread.spawn(.{}, Rank.runCore, .{rk[1]});
    const g1 = try std.Thread.spawn(.{}, FakeGpu.run, .{&g[1]});
    g[0].run();
    g1.join();
    for (rk) |r| r.stop.store(1, .release);
    c0.join();
    c1.join();
    for (s) |th| if (th) |t| t.join();
    try testing.expectEqual(@as(usize, 0), g[0].bad + g[1].bad);
    try testing.expectEqual(@as(u64, 2000), rk[0].exchanged.load(.acquire));
    for (rk) |r| {
        try testing.expectEqual(@as(u32, 0), r.code);
        try testing.expectEqual(@as(u32, 0), r.code2);
    }
    if (stripe_min) |m| {
        // each rank's second wire moved the second half of every gather of at least m bytes, and nothing else
        var split: u64 = 0;
        for (1..2001) |q| split += @intFromBool(1000 + (q % 7) * 4096 >= m);
        try testing.expect(split > 0 and split < 2000);
        try testing.expectEqual(split, rk[0].w2.?.seq);
        try testing.expectEqual(split, rk[1].w2.?.seq);
    }
}

test "a slot is not reused before it was consumed: a stalled collect stops the stage, not the data" {
    const gpa = testing.allocator;
    const rk = try rankPair(gpa, null);
    defer for (rk) |r| {
        r.close();
        gpa.destroy(r);
    };
    var g: [2]FakeGpu = undefined;
    for (&g, rk, 0..) |*f, r, i| f.* = .{ .shared = &r.shared, .tx = &r.tx, .rx = &r.rx, .nslot = 4, .slot_bytes = 32768, .rank = @intCast(i), .gathers = 50 };
    g[1].hold_collect_from = 10; // rank 1 stops collecting at gather 10: its stage must stall at gather 14
    const c0 = try std.Thread.spawn(.{}, Rank.runCore, .{rk[0]});
    const c1 = try std.Thread.spawn(.{}, Rank.runCore, .{rk[1]});
    const g0 = try std.Thread.spawn(.{}, FakeGpu.run, .{&g[0]});
    g[1].run();
    try testing.expect(g[1].timed_out); // waited for a slot that was never released
    for (rk) |r| {
        r.stop.store(1, .release);
        r.w.abort();
    }
    @atomicStore(u32, &rk[0].shared.abort, 1, .release);
    g0.join();
    c0.join();
    c1.join();
    try testing.expectEqual(@as(usize, 0), g[0].bad + g[1].bad); // nothing overwritten while still owned
    try testing.expect(rk[1].exchanged.load(.acquire) <= 13);
}

test "a generation ahead of the expected one is refused and raises the abort word" {
    const gpa = testing.allocator;
    const rk = try rankPair(gpa, null);
    defer for (rk) |r| {
        r.close();
        gpa.destroy(r);
    };
    @atomicStore(u64, &rk[0].shared.slot[1].ready, 5, .release); // gather 5 published where gather 1 belongs
    rk[0].runCore();
    try testing.expectEqual(@as(u32, 3), rk[0].code);
    try testing.expect(@atomicLoad(u32, &rk[0].shared.abort, .acquire) != 0);
}

test "the device waits' deadline in wall-clock ticks: 600 s at 100 MHz does not overflow" {
    try testing.expectEqual(@as(u64, 1_000_000_000), deadlineTicks(100_000_000, 10 * std.time.ns_per_s));
    try testing.expectEqual(@as(u64, 6_000_000_000), deadlineTicks(100_000_000, 60 * std.time.ns_per_s));
    try testing.expectEqual(@as(u64, 60_000_000_000), deadlineTicks(100_000_000, 600 * std.time.ns_per_s)); // 6e19 as a u64 product
}

test "a device wait that gave up (abort word 2) fails the proxy with code 2, marked logged; the host's 1 never overwrites it" {
    const gpa = testing.allocator;
    const rk = try rankPair(gpa, null);
    defer for (rk) |r| {
        r.close();
        gpa.destroy(r);
    };
    @atomicStore(u32, &rk[0].shared.abort, 2, .release); // coll.hip tf_ds_coll_fail
    rk[0].runCore();
    try testing.expectEqual(@as(u32, 2), rk[0].code);
    try testing.expectEqual(@as(u32, 3), @atomicLoad(u32, &rk[0].shared.abort, .acquire)); // its line logged
    raiseAbort(&rk[0].shared);
    try testing.expectEqual(@as(u32, 3), @atomicLoad(u32, &rk[0].shared.abort, .acquire));
    raiseAbort(&rk[1].shared);
    try testing.expectEqual(@as(u32, 1), @atomicLoad(u32, &rk[1].shared.abort, .acquire));
}

test "a peer that stops staging fails the proxy's exchange at the progress deadline (Timeout), not at a GPU wait's" {
    const gpa = testing.allocator;
    const rk = try rankPair(gpa, null); // progress deadline 500 ms
    defer for (rk) |r| {
        r.close();
        gpa.destroy(r);
    };
    // rank 0 stages gather 1; rank 1 never does
    rk[0].shared.slot[1].bytes = 4096;
    rk[0].shared.slot[1].site = (@as(u32, @intFromEnum(coll.Phase.prompt_engram)) << 16) | 3;
    @atomicStore(u64, &rk[0].shared.slot[1].ready, 1, .release);
    var core = rk[0].core();
    core.warn_ns = 100 * std.time.ns_per_ms;
    const t0 = wire.monotonicNs();
    const code = core.run();
    const waited = wire.monotonicNs() - t0;
    try testing.expectEqual(@as(u32, 2), code);
    try testing.expect(rk[0].w.status().? == error.Timeout);
    try testing.expect(waited >= 500 * std.time.ns_per_ms and waited < 5 * std.time.ns_per_s);
    try testing.expectEqual(@as(u32, 1), @atomicLoad(u32, &rk[0].shared.abort, .acquire));
}
