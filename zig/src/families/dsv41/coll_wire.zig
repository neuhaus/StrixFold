//! Payload exchange for eager HIP gathers over TCP or registered host memory.
const std = @import("std");
const wire = @import("tcp_wire.zig");

pub const HostWire = union(enum) {
    tcp: wire.Wire,
    verbs: *@import("verbs_wire.zig").Wire,

    pub fn abort(w: *HostWire) void {
        switch (w.*) {
            .tcp => |*t| t.abort(),
            .verbs => |v| v.abort(),
        }
    }

    pub fn close(w: *HostWire, gpa: std.mem.Allocator) void {
        switch (w.*) {
            .tcp => |*t| t.close(std.time.ns_per_s),
            .verbs => |v| v.close(gpa),
        }
    }

    pub fn status(w: *const HostWire) ?wire.Error {
        return switch (w.*) {
            .tcp => |*t| t.status(),
            .verbs => |v| v.status(),
        };
    }

    pub fn seq(w: *const HostWire) u64 {
        return switch (w.*) {
            .tcp => |*t| t.seq,
            .verbs => |v| v.seq,
        };
    }

    pub fn detail(w: *const HostWire) []const u8 {
        return switch (w.*) {
            .tcp => |*t| std.mem.sliceTo(&t.why, 0),
            .verbs => "RC verbs",
        };
    }

    pub fn exchange(w: *HostWire, site: u32, out: []const u8, in: []u8) wire.Error!void {
        return switch (w.*) {
            .tcp => |*t| t.exchange(site, out, in),
            .verbs => |v| v.exchange(site, out, in),
        };
    }
};
