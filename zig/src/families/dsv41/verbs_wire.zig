//! Registered host windows for HIP TP2; each receive credit authorizes one RC WRITE_WITH_IMM.
const std = @import("std");
const fabric = @import("fabric");
const tcp = @import("tcp_wire.zig");
const Link = @import("link.zig").Link;
const abi = fabric.verbs_abi;
const Endpoint = fabric.verbs_qp.Endpoint;
const Error = tcp.Error;
const Header = tcp.Header;
const header_bytes = @sizeOf(Header);
const depth = 8;
const wr_send_imm = 3;
const wc_recv = 128;
const wc_recv_write_imm = 129;
const wc_send = 0;
const wc_write = 1;
const tx_credit = 1;
const tx_payload = 2;
const rx_notify = 3;
pub const version = 1;

pub const Wire = struct {
    v: fabric.verbs.Verbs,
    ep: Endpoint,
    tx: []align(std.heap.page_size_min) u8,
    rx: []align(std.heap.page_size_min) u8,
    tx_mr: *abi.Mr,
    rx_mr: *abi.Mr,
    remote: Info,
    epoch: [16]u8,
    rank: u16,
    progress_ns: u64,
    seq: u64 = 0,
    aborted: std.atomic.Value(bool) = .init(false),
    /// The exchange thread owns the QP. Other threads can abort it without touching verbs.
    failed: std.atomic.Value(u16) = .init(0),

    pub fn open(gpa: std.mem.Allocator, link: Link, device: []const u8, gid_index: u8, rank: u16, epoch: [16]u8, capacity: usize, progress_ns: u64) !*Wire {
        const w = try gpa.create(Wire);
        errdefer gpa.destroy(w);
        w.v = try fabric.verbs.Verbs.open(null);
        errdefer w.v.close();
        if (capacity == 0 or rank > 1) return error.BadSetting;
        const len = try std.math.add(usize, capacity, header_bytes);
        const prot: std.posix.PROT = .{ .READ = true, .WRITE = true };
        const flags: std.posix.MAP = .{ .TYPE = .PRIVATE, .ANONYMOUS = true };
        w.tx = try std.posix.mmap(null, len, prot, flags, -1, 0);
        errdefer std.posix.munmap(w.tx);
        w.rx = try std.posix.mmap(null, len, prot, flags, -1, 0);
        errdefer std.posix.munmap(w.rx);
        w.ep = try Endpoint.open(&w.v, device, .rc, depth, gid_index);
        // QP destruction must precede deregistration, and its PD must outlive the MRs.
        var tx_mr: ?*abi.Mr = null;
        var rx_mr: ?*abi.Mr = null;
        errdefer w.releaseEndpoint(tx_mr, rx_mr);
        const local = try w.ep.info();
        if (std.mem.allEqual(u8, &local.gid, 0) or w.ep.port.active_mtu < 1 or w.ep.port.active_mtu > 5 or
            len > w.ep.port.max_msg_sz) return error.BadVerbsPort;
        w.tx_mr = w.v.api.ibv_reg_mr(w.ep.pd, w.tx.ptr, len, 0) orelse return error.VerbsTxRegistration;
        tx_mr = w.tx_mr;
        w.rx_mr = try w.ep.register(w.rx);
        rx_mr = w.rx_mr;

        const mine = (Info{
            .qpn = local.qpn,
            .psn = local.psn,
            .lid = local.lid,
            .mtu = @intCast(w.ep.port.active_mtu),
            .gid = local.gid,
            .addr = @intFromPtr(w.rx.ptr),
            .rkey = w.rx_mr.rkey,
        }).encode();
        try link.send(&mine);
        var peer_buf: [Info.len]u8 = undefined;
        w.remote = try Info.decode(try link.recv(&peer_buf), len);
        try w.ep.connect(.{
            .qpn = w.remote.qpn,
            .psn = w.remote.psn,
            .lid = w.remote.lid,
            .gid = w.remote.gid,
            .gid_index = 0,
        }, @min(w.ep.port.active_mtu, w.remote.mtu), .{});
        for (0..depth) |_| try w.ep.receive(&.{}, 0, rx_notify);
        try link.send(&.{version});
        var ready: [1]u8 = undefined;
        if (!std.mem.eql(u8, try link.recv(&ready), &.{version})) return error.Mismatch;
        w.epoch = epoch;
        w.rank = rank;
        w.progress_ns = progress_ns;
        w.seq = 0;
        w.aborted = .init(false);
        w.failed = .init(0);
        std.log.info("collectives: RC verbs on {s} GID index {d}, MTU {d}, registered host window {d} bytes", .{
            device, gid_index, @as(u32, 128) << @intCast(@min(w.ep.port.active_mtu, w.remote.mtu)), len,
        });
        return w;
    }

    fn releaseEndpoint(w: *Wire, tx_mr: ?*abi.Mr, rx_mr: ?*abi.Mr) void {
        const api = w.v.api;
        _ = api.ibv_destroy_qp(w.ep.qp);
        if (rx_mr) |mr| _ = api.ibv_dereg_mr(mr);
        if (tx_mr) |mr| _ = api.ibv_dereg_mr(mr);
        _ = api.ibv_destroy_cq(w.ep.rcq);
        _ = api.ibv_destroy_cq(w.ep.scq);
        _ = api.ibv_dealloc_pd(w.ep.pd);
        _ = api.ibv_close_device(w.ep.ctx);
    }

    pub fn abort(w: *Wire) void {
        w.aborted.store(true, .release);
    }

    /// Call after the owning exchange thread has stopped.
    pub fn close(w: *Wire, gpa: std.mem.Allocator) void {
        // Destroy the QP first to stop DMA to the RX window before unregistering it.
        w.releaseEndpoint(w.tx_mr, w.rx_mr);
        w.v.close();
        std.posix.munmap(w.rx);
        std.posix.munmap(w.tx);
        gpa.destroy(w);
    }

    pub fn status(w: *const Wire) ?Error {
        const failure = w.failed.load(.acquire);
        if (failure != 0) return @errorCast(@errorFromInt(failure));
        return if (w.aborted.load(.acquire)) error.Aborted else null;
    }

    pub fn exchange(w: *Wire, site: u32, out: []const u8, in: []u8) Error!void {
        if (w.status()) |e| return e;
        w.run(site, out, in) catch |e| {
            w.failed.store(@intFromError(e), .release);
            return e;
        };
    }

    fn run(w: *Wire, site: u32, out: []const u8, in: []u8) Error!void {
        if (out.len != in.len) return error.Mismatch;
        const capacity = w.tx.len - header_bytes;
        const count = @max(1, std.math.divCeil(usize, out.len, capacity) catch unreachable);
        const parts = std.math.cast(u8, count) orelse return error.Mismatch;
        for (0..parts) |i| {
            const offset = i * capacity;
            const n = @min(out.len - offset, capacity);
            try w.fragment(site, out[offset..][0..n], in[offset..][0..n], out.len, @intCast(i), parts);
        }
    }

    fn fragment(w: *Wire, site: u32, out: []const u8, in: []u8, bytes: usize, part: u8, parts: u8) Error!void {
        try w.checkAbort();
        w.seq += 1;
        const mine: Header = .{
            .magic = tcp.magic,
            .version = version,
            .flags = 0,
            .epoch = w.epoch,
            .seq = w.seq,
            .bytes = bytes,
            .site = site,
            .rank = w.rank,
            .part = part,
            .parts = parts,
        };
        @memcpy(w.tx[0..header_bytes], std.mem.asBytes(&mine));
        @memcpy(w.tx[header_bytes..][0..out.len], out);
        const imm = std.mem.nativeToBig(u32, @truncate(w.seq));
        var sge: abi.Sge = .{ .addr = @intFromPtr(w.tx.ptr), .length = @intCast(header_bytes + out.len), .lkey = w.tx_mr.lkey };
        try w.post(.{ .wr_id = tx_credit, .sg_list = @ptrCast(&sge), .num_sge = 0, .opcode = wr_send_imm, .send_flags = abi.send_signaled, .imm_data = imm });
        var last = tcp.monotonicNs();
        var credit_sent = false;
        var write_posted = false;
        var write_sent = false;
        var write_received = false;
        while (!write_sent or !write_received or !credit_sent) {
            try w.checkAbort();
            var moved = false;
            if (try w.poll(w.ep.scq)) |wc| {
                if (wc.wr_id == tx_credit and wc.opcode == wc_send and !credit_sent) {
                    credit_sent = true;
                } else if (wc.wr_id == tx_payload and wc.opcode == wc_write and write_posted and !write_sent) {
                    write_sent = true;
                } else return error.Mismatch;
                moved = true;
            }
            if (!write_received) if (try w.poll(w.ep.rcq)) |wc| {
                if (wc.wr_id != rx_notify or wc.wc_flags & abi.wc_with_imm == 0 or wc.imm_data != imm) return error.Mismatch;
                w.ep.receive(&.{}, 0, rx_notify) catch return error.SocketFailed;
                if (wc.opcode == wc_recv and !write_posted and wc.byte_len == 0) {
                    try w.post(.{
                        .wr_id = tx_payload,
                        .sg_list = @ptrCast(&sge),
                        .num_sge = 1,
                        .opcode = abi.wr_rdma_write_imm,
                        .send_flags = abi.send_signaled,
                        .imm_data = imm,
                        .remote_addr = w.remote.addr,
                        .rkey = w.remote.rkey,
                    });
                    write_posted = true;
                } else if (wc.opcode == wc_recv_write_imm and write_posted and wc.byte_len == sge.length) {
                    var theirs: Header = undefined;
                    @memcpy(std.mem.asBytes(&theirs), w.rx[0..header_bytes]);
                    if (!matches(mine, theirs)) return error.Mismatch;
                    @memcpy(in, w.rx[header_bytes..][0..in.len]);
                    write_received = true;
                } else return error.Mismatch;
                moved = true;
            };
            if (moved) last = tcp.monotonicNs() else {
                if (tcp.monotonicNs() -| last >= w.progress_ns) return error.Timeout;
                std.atomic.spinLoopHint();
            }
        }
    }

    fn checkAbort(w: *const Wire) Error!void {
        if (w.aborted.load(.acquire)) return error.Aborted;
    }

    fn post(w: *const Wire, request: abi.SendWr) Error!void {
        var wr = request;
        var bad: ?*abi.SendWr = null;
        if (w.ep.ctx.ops.post_send(w.ep.qp, &wr, &bad) != 0) return error.SocketFailed;
    }

    fn poll(w: *const Wire, cq: *abi.Cq) Error!?abi.Wc {
        var wc: [1]abi.Wc = undefined;
        const n = w.ep.ctx.ops.poll_cq(cq, 1, &wc);
        if (n == 0) return null;
        if (n < 0) return error.SocketFailed;
        if (wc[0].status == abi.wc_success) return wc[0];
        std.log.warn("collectives: verbs rank {d} gather {d}, CQ status {s} ({d}), vendor {d}", .{
            w.rank, w.seq, w.v.statusText(wc[0].status), wc[0].status, wc[0].vendor_err,
        });
        // RETRY_EXC_ERR / RNR_RETRY_EXC_ERR indicate a peer that stopped progressing.
        return if (wc[0].status == 12 or wc[0].status == 13) error.PeerLost else error.SocketFailed;
    }
};

fn matches(mine: Header, peer: Header) bool {
    return peer.magic == mine.magic and peer.version == mine.version and peer.flags == 0 and
        std.mem.eql(u8, &peer.epoch, &mine.epoch) and peer.seq == mine.seq and peer.bytes == mine.bytes and
        peer.site == mine.site and peer.rank == 1 - mine.rank and peer.part == mine.part and peer.parts == mine.parts;
}

const Info = struct {
    qpn: u32,
    psn: u32,
    lid: u16,
    mtu: u8,
    gid: [16]u8,
    addr: u64,
    rkey: u32,

    const len = 40;

    fn encode(i: Info) [len]u8 {
        var b: [len]u8 = @splat(0);
        std.mem.writeInt(u32, b[0..4], i.qpn, .little);
        std.mem.writeInt(u32, b[4..8], i.psn, .little);
        std.mem.writeInt(u16, b[8..10], i.lid, .little);
        b[10] = i.mtu;
        b[11] = version;
        @memcpy(b[12..28], &i.gid);
        std.mem.writeInt(u64, b[28..36], i.addr, .little);
        std.mem.writeInt(u32, b[36..40], i.rkey, .little);
        return b;
    }

    fn decode(b: []const u8, capacity: usize) !Info {
        if (b.len != len or b[11] != version or b[10] < 1 or b[10] > 5) return error.Mismatch;
        const i: Info = .{
            .qpn = std.mem.readInt(u32, b[0..4], .little),
            .psn = std.mem.readInt(u32, b[4..8], .little),
            .lid = std.mem.readInt(u16, b[8..10], .little),
            .mtu = b[10],
            .gid = b[12..28].*,
            .addr = std.mem.readInt(u64, b[28..36], .little),
            .rkey = std.mem.readInt(u32, b[36..40], .little),
        };
        if (i.qpn == 0 or i.qpn > 0xffffff or i.psn > 0xffffff or std.mem.allEqual(u8, &i.gid, 0) or
            i.addr == 0 or i.addr > std.math.maxInt(u64) - capacity) return error.Mismatch;
        return i;
    }
};

test "verbs bootstrap rejects a corrupt or overflowing peer window" {
    const info: Info = .{ .qpn = 9, .psn = 0xfffffe, .lid = 0, .mtu = 5, .gid = @splat(1), .addr = 4096, .rkey = 0 };
    const encoded = info.encode();
    try std.testing.expectEqual(info, try Info.decode(&encoded, 4096));
    try std.testing.expectError(error.Mismatch, Info.decode(encoded[0..39], 4096));
    var bad = encoded;
    bad[10] = 6;
    try std.testing.expectError(error.Mismatch, Info.decode(&bad, 4096));
    bad = encoded;
    std.mem.writeInt(u64, bad[28..36], std.math.maxInt(u64) - 63, .little);
    try std.testing.expectError(error.Mismatch, Info.decode(&bad, 4096));
}

// TF_VERBS_TEST_RANK opts in to the two-host test without a HIP runtime or model files.
test "verbs pair: checked bidirectional payloads, window reuse and terminal failure" {
    const rank_text = std.c.getenv("TF_VERBS_TEST_RANK") orelse return error.SkipZigTest;
    const rank = try std.fmt.parseInt(u16, std.mem.span(rank_text), 10);
    if (rank > 1) return error.BadSetting;
    const device = std.mem.span(std.c.getenv("TF_COLL_VERBS_DEVICE") orelse return error.MissingVerbsDevice);
    const ip = try @import("link.zig").parseIp(std.mem.span(std.c.getenv("TF_VERBS_TEST_MASTER") orelse return error.BadAddress));
    const port = try std.fmt.parseInt(u16, std.mem.span(std.c.getenv("TF_VERBS_TEST_PORT") orelse return error.BadAddress), 10);
    const gid = if (std.c.getenv("TF_COLL_VERBS_GID_INDEX")) |p| try std.fmt.parseInt(u8, std.mem.span(p), 10) else 1;
    const mode = if (std.c.getenv("TF_VERBS_TEST_FAILURE")) |p| std.mem.span(p) else "";
    const a = std.testing.allocator;
    var link = if (rank == 0) try Link.listen(ip, port, 10) else try Link.connect(std.testing.io, ip, port, 10);
    defer link.close();
    const deadline = 500 * std.time.ns_per_ms;
    const w = try Wire.open(a, link, device, gid, rank, @splat(3), 4 << 20, deadline);
    defer w.close(a);
    if (std.mem.eql(u8, mode, "peer-exit")) {
        if (rank == 1) return;
        const start = tcp.monotonicNs();
        var out: [1]u8 = undefined;
        const failure = if (w.exchange(1, "x", &out)) |_| return error.ExpectedFailure else |e| e;
        try std.testing.expect(failure == error.PeerLost or failure == error.Timeout);
        try std.testing.expect(tcp.monotonicNs() - start < 2 * std.time.ns_per_s);
        try std.testing.expectError(failure, w.exchange(1, "x", &out));
        return;
    }
    if (std.mem.eql(u8, mode, "mismatch")) {
        var out: [1]u8 = undefined;
        try std.testing.expectError(error.Mismatch, w.exchange(rank, "x", &out));
        try std.testing.expectError(error.Mismatch, w.exchange(0, "x", &out));
        return;
    }
    const sizes = [_]usize{ 20 << 10, 5, 120 << 10, 64 << 10, 4 << 20, (8 << 20) + 11 };
    const send = try a.alloc(u8, sizes[sizes.len - 1]);
    defer a.free(send);
    const recv = try a.alloc(u8, send.len);
    defer a.free(recv);
    for (0..24) |i| {
        const n = sizes[i % sizes.len];
        const seed: u8 = @truncate(i * 23);
        const mine = seed +% @as(u8, @intCast(rank * 83));
        const peer = seed +% @as(u8, @intCast((1 - rank) * 83));
        for (send[0..n], 0..) |*b, off| b.* = mine +% @as(u8, @truncate(off ^ (off >> 11)));
        @memset(recv[0..n], 0);
        try w.exchange(@intCast(i), send[0..n], recv[0..n]);
        for (recv[0..n], 0..) |b, off|
            try std.testing.expectEqual(peer +% @as(u8, @truncate(off ^ (off >> 11))), b);
    }
}
