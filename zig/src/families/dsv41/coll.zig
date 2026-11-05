//! The TP ranks' collectives as the family calls them (tools/dsv41-hip/ADAPTER.md section 5; the interface agreed with
//! w-link on 2026-10-19). Every model reduction is an all-gather of fp32 partials that each rank then adds in rank order
//! (DESIGN.md section 3), so a collective here only moves bytes: a stream-ordered all-gather in rank order, the
//! four-node split's point-to-point exchange (CUDA's NCCL only), and abort. NCCL implements it on CUDA (comm.zig
//! Comm.coll); w-link's host-staged TCP transport on HIP, handed to model.open as Options.open_coll.
//!
//! The contract an implementation keeps:
//! - Every rank issues the same sequence of calls: the same site and byte count in the same order. A transport that
//!   carries a header (TCP) checks the site, its layer, the bytes and its own generation against the peer's and fails
//!   the call with Mismatch on any difference; NCCL carries none and does not check.
//! - The first failure poisons the Coll: that call and every later one return the same error.
//! - abort is idempotent and may be called from any thread; calls blocked in the transport return (Aborted, or the
//!   error that poisoned it first).
//! - close releases everything in a bounded time; a call still in flight is aborted first.
//! - Deadlines (progress-based) and generations are the implementation's own; they are not parameters.
const std = @import("std");
const gpu = @import("gpu");
const Fan = @import("fan.zig").Fan;

/// The element types the family moves (the names NCCL's DataType gives them); a collective only counts their bytes.
pub const DType = enum {
    i8,
    u8,
    i32,
    u32,
    i64,
    u64,
    f16,
    f32,
    f64,
    bf16,

    pub fn size(t: DType) usize {
        return switch (t) {
            .i8, .u8 => 1,
            .f16, .bf16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64 => 8,
        };
    }
};

/// A collective's failure, the same on every later call (the Coll is poisoned).
pub const Error = error{
    /// the transport failed (a socket or device copy error, NCCL's error code)
    CollectiveFailed,
    /// the peer closed or reset its connection
    PeerLost,
    /// no progress within the implementation's deadline
    Timeout,
    /// the peer issued a different call (site, layer, bytes or generation)
    Mismatch,
    /// abort was called
    Aborted,
};

/// Why a collective stopped: what a rank tells its peer and its client.
pub const Reason = enum { cancelled, peer_lost, timeout, device_fault, shutdown };

/// Where in the forward a collective is issued: every all-gather the family makes, by pass. The decode rounds' and the
/// drafter's are the gathers among round.Point / draft.Point; `layer` is the layer (the drafter's block, the Markov
/// step; the head's is the layer count).
pub const Phase = enum(u16) {
    round_engram = 1, // round.zig stretch: Point.engram_gather
    round_attn, // Point.attn_gather
    round_moe, // Point.moe_gather
    round_head, // Point.head_gather
    prompt_engram, // prompt.zig engramApply (a prompt chunk)
    prompt_attn, // prompt.gather after attention
    prompt_moe, // prompt.gather after the MoE
    prompt_head, // prompt.head
    draft_attn, // draft.zig body: draft.Point.attn_gather
    draft_moe, // draft.Point.moe_gather
    draft_markov, // the Markov steps' bests (layer: the step)
    ring_setup, // CUDA: the RDMA rings' queue-pair infos and their start barrier
    two_parts, // CUDA world 4: prompt2d's parts
    two_pair, // CUDA world 4: prompt2d's pair exchanges
    test_only = 0xffff,
};

pub const Site = struct {
    phase: Phase,
    layer: u16,

    pub fn at(phase: Phase, layer: usize) Site {
        return .{ .phase = phase, .layer = @intCast(layer) };
    }
};

pub fn controlFrameBytes(pool: usize) usize {
    return (pool + 1) * @sizeOf(i32) + (1 << 16);
}

pub const Control = struct {
    ptr: *anyopaque,
    send: *const fn (*anyopaque, []const u8) Error!void,
    recv: *const fn (*anyopaque, []u8, u64) Error![]const u8,
};

pub const Coll = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    rank: u32,
    world: u32,
    /// Its all-gather may be captured into a graph (NCCL, or a device-proxy transport); a host-staged transport is not,
    /// so the rounds run eagerly with it (TF_DS_GRAPHS=1 is refused at open).
    capturable: bool,
    control: ?Control = null,

    pub const VTable = struct {
        /// recv[r * bytes .. (r + 1) * bytes] = rank r's send, for every rank in rank order. Stream-ordered: reads `send`
        /// after the work already queued on `stream`; work queued after it sees `recv`. A host-staged transport may block
        /// the calling thread until `recv` is written.
        all_gather: *const fn (ptr: *anyopaque, stream: gpu.Stream, site: Site, send: u64, recv: u64, bytes: usize) Error!void,
        /// `send_bytes` to rank `peer` and `recv_bytes` from it, stream-ordered (the four-node split; null: none).
        exchange: ?*const fn (ptr: *anyopaque, stream: gpu.Stream, site: Site, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, peer: u32) Error!void = null,
        /// Fails every call in flight and after with Aborted (unless already poisoned); idempotent, any thread.
        abort: *const fn (ptr: *anyopaque, reason: Reason) void,
        /// Releases the transport in a bounded time, aborting a call in flight.
        close: *const fn (ptr: *anyopaque) void,
        /// The all-gather in two halves (Coll.gatherBegin / gatherEnd; null: the gather runs whole at its end).
        gather_begin: ?*const fn (ptr: *anyopaque, stream: gpu.Stream, site: Site, send: u64, recv: u64, bytes: usize) Error!void = null,
        gather_end: ?*const fn (ptr: *anyopaque, stream: gpu.Stream) Error!void = null,
        /// The transport's failure, if it has failed (null: it reports failures only from its calls). A transport whose
        /// gathers run without a host call (a device proxy, a captured graph) is checked after the stream synchronizes.
        check: ?*const fn (ptr: *anyopaque) Error!void = null,
        /// How many gathers this rank's transport has moved so far (diagnostics; null: it does not count).
        counts: ?*const fn (ptr: *anyopaque) Counts = null,
    };

    /// A transport's running totals, equal on both ranks whenever their streams stand at the same point: device-proxy
    /// gathers staged (tcp_proxy's slab sequence) and the exchanges of each of its wires (both proxy cables, tcp_host's).
    pub const Counts = struct { staged: u64 = 0, cable1: u64 = 0, cable2: u64 = 0, host: u64 = 0 };

    pub fn counts(c: Coll) Counts {
        return if (c.vtable.counts) |f| f(c.ptr) else .{};
    }

    /// After the stream synchronized: fails with the transport's error when a gather the host did not call (a captured
    /// one, a device wait that gave up at its deadline) failed, so no result of that work is used.
    pub fn check(c: Coll) Error!void {
        if (c.vtable.check) |f| return f(c.ptr);
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank, at `site`.
    pub fn allGather(c: Coll, site: Site, send: u64, recv: u64, count: usize, dtype: DType, stream: gpu.Stream) Error!void {
        return c.vtable.all_gather(c.ptr, stream, site, send, recv, count * dtype.size());
    }

    /// A begun all-gather (gatherBegin), finished by gatherEnd.
    pub const Pending = struct { site: Site, send: u64, recv: u64, bytes: usize };

    /// allGather split so the bytes can move while the GPU runs other work: gatherBegin takes `send` as the work
    /// queued on `stream` so far leaves it and returns without waiting for it; work queued between the two calls must
    /// neither write `send` nor touch `recv`; gatherEnd completes the gather (a host-staged transport blocks there
    /// while the bytes move), and work queued on `stream` after it sees `recv`. One begun gather at a time per Coll and
    /// no other call on it in between; both ranks pair them alike. A transport without the split gathers at the end
    /// (the same bytes, no overlap).
    pub fn gatherBegin(c: Coll, site: Site, send: u64, recv: u64, count: usize, dtype: DType, stream: gpu.Stream) Error!Pending {
        const p: Pending = .{ .site = site, .send = send, .recv = recv, .bytes = count * dtype.size() };
        if (c.vtable.gather_begin) |f| try f(c.ptr, stream, site, send, recv, p.bytes);
        return p;
    }

    pub fn gatherEnd(c: Coll, p: Pending, stream: gpu.Stream) Error!void {
        if (c.vtable.gather_end) |f| return f(c.ptr, stream);
        return c.vtable.all_gather(c.ptr, stream, p.site, p.send, p.recv, p.bytes);
    }

    /// `count` elements to rank `peer` and `rcount` from it (the four-node split's pairs).
    pub fn exchange(c: Coll, site: Site, send: u64, count: usize, recv: u64, rcount: usize, dtype: DType, peer: u32, stream: gpu.Stream) (Error || error{Unsupported})!void {
        const f = c.vtable.exchange orelse return error.Unsupported;
        return f(c.ptr, stream, site, send, count * dtype.size(), recv, rcount * dtype.size(), peer);
    }

    pub fn abort(c: Coll, reason: Reason) void {
        c.vtable.abort(c.ptr, reason);
    }

    pub fn close(c: Coll) void {
        c.vtable.close(c.ptr);
    }
};

/// What a rank brings to the peers' handshake (w-link's, inside Open): the transport compares it with every other rank's
/// and refuses the open on a difference, naming the field. `plan` is text (space-separated key=value): the settings that
/// decide which collectives a rank issues and with what sizes, so ranks that would issue different sequences never
/// start (model.zig Model.identity writes it).
pub const Identity = struct {
    rank: u32,
    world: u32,
    /// abi.h TF_DS_HIP_ABI_VERSION this binary was built for (0 on CUDA: no AMD kit)
    kit_abi: u32,
    /// SHA-256 (hex) of the AMD kit's manifest.json: the same code objects on every rank (zeros on CUDA)
    kit_sha256: [64]u8,
    /// model.Modalities as its tag (text 0, text_image 1)
    modalities: u8,
    plan: []const u8,
    control_bytes: usize = 0,
};

/// Opens a rank's collectives over its links (handshake first): the transport a HIP build plugs in. Errors here are
/// the transport's own (a refused handshake is Mismatch); the identity is the caller's and outlives the call only.
/// What a device-side transport binds its kernels from (tcp_proxy's coll.stage / coll.collect): the HIP build's AMD
/// kit; CUDA builds have none. The kit outlives the Coll (model.deinit closes the collective first).
pub const Kit = if (gpu.backend == .hip) gpu.Kit else void;
pub const Open = *const fn (gpa: std.mem.Allocator, io: std.Io, ctx: *const gpu.Context, kit: ?*const Kit, fan: *const Fan, id: *const Identity) anyerror!Coll;

test "a collective counts bytes by element type, passes its site, and refuses an exchange it lacks" {
    const Fake = struct {
        bytes: usize = 0,
        site: Site = .{ .phase = .test_only, .layer = 0 },
        fn gather(ptr: *anyopaque, _: gpu.Stream, site: Site, _: u64, _: u64, bytes: usize) Error!void {
            const f: *@This() = @ptrCast(@alignCast(ptr));
            f.bytes = bytes;
            f.site = site;
        }
        fn abort(_: *anyopaque, _: Reason) void {}
        fn close(_: *anyopaque) void {}
    };
    var f: Fake = .{};
    const vt: Coll.VTable = .{ .all_gather = Fake.gather, .abort = Fake.abort, .close = Fake.close };
    const c: Coll = .{ .ptr = &f, .vtable = &vt, .rank = 0, .world = 2, .capturable = false };
    try c.allGather(.at(.round_attn, 42), 0, 0, 6 * 5120, .f32, undefined);
    try std.testing.expectEqual(@as(usize, 6 * 5120 * 4), f.bytes);
    try std.testing.expectEqual(Phase.round_attn, f.site.phase);
    try std.testing.expectEqual(@as(u16, 42), f.site.layer);
    try c.allGather(.at(.round_head, 61), 0, 0, 7, .bf16, undefined);
    try std.testing.expectEqual(@as(usize, 14), f.bytes);
    try std.testing.expectError(error.Unsupported, c.exchange(.at(.two_pair, 0), 0, 1, 0, 1, .f32, 1, undefined));
    // a transport without the split gathers whole at gatherEnd
    const p = try c.gatherBegin(.at(.prompt_moe, 3), 0, 0, 10, .f32, undefined);
    try std.testing.expectEqual(@as(usize, 14), f.bytes);
    try c.gatherEnd(p, undefined);
    try std.testing.expectEqual(@as(usize, 40), f.bytes);
    try std.testing.expectEqual(Phase.prompt_moe, f.site.phase);
}
