//! Compile and test the dsv41 HIP serving path without requiring a GPU.
const std = @import("std");
const gpu = @import("gpu");
const native = @import("native.zig");
const model = @import("model.zig");

test "a HIP rank's serving path compiles without CUDA, NCCL, the CUDA RDMA ring or vision" {
    comptime std.debug.assert(gpu.backend == .hip and !gpu.features.vision and !gpu.features.verbs_rdma and !gpu.features.world4);
    _ = &native.open;
    _ = &native.explain;
    _ = &native.weightBytes;
    _ = &native.modalities;
    _ = &model.Model.open;
    _ = &model.Model.close;
}

test "the text capability is a HIP rank's default and image input is refused by name" {
    try std.testing.expectEqual(model.Modalities.text, model.default_modalities);
    try std.testing.expectEqual(model.Modalities.text, try native.modalities(.{ .context = 8192, .drafts = true }));
    try std.testing.expectEqual(model.Modalities.text_image, try native.modalities(.{ .context = 8192, .drafts = true, .modalities = .text_image }));
    try std.testing.expect(std.mem.indexOf(u8, native.explain(null, error.VisionUnsupported).?, "text only") != null);
    try std.testing.expect(std.mem.indexOf(u8, native.explain(null, error.NoCollective).?, "w-link") != null);
}

test {
    // the host tests of the files a HIP rank uses
    _ = @import("coll.zig");
    _ = @import("model.zig");
    _ = @import("tcp_wire.zig");
    _ = @import("tcp_coll.zig");
    _ = @import("tcp_proxy.zig");
    _ = @import("verbs_wire.zig");
    _ = @import("fan.zig");
    _ = @import("config.zig");
    _ = @import("plan.zig");
    _ = @import("rank_cache.zig");
    _ = @import("checkpoint.zig");
    _ = @import("engram.zig");
    _ = @import("engram_io.zig");
    _ = @import("engram_aio.zig");
    _ = @import("sampling.zig");
    _ = @import("text.zig");
    _ = @import("markov_tokens.zig");
    _ = @import("tri.zig");
    _ = @import("tri_basic.zig");
    _ = @import("tri_hc.zig");
    _ = @import("tri_norm.zig");
    _ = @import("tri_index.zig");
    _ = @import("tri_attn.zig");
    _ = @import("tri_aux.zig");
    _ = @import("tri_markov.zig");
    _ = @import("ops.zig");
    _ = @import("exact.zig");
    _ = @import("weights.zig");
    _ = @import("draft.zig");
    _ = @import("round.zig");
    _ = @import("prompt.zig");
    _ = @import("lanes.zig");
    _ = @import("native.zig");
    _ = @import("hipblas.zig");
    _ = @import("hip_load.zig");
}
