//! The HIP half of the root build for the dsv41 port (tools/dsv41-hip/ADAPTER.md section 9): the `gpu` module for HIP
//! and CUDA, host tests and GPU tests. Nothing in a HIP module's graph imports zig/src/cuda; the CUDA targets in cuda.zig
//! are untouched. Options (the gfx1151 receipt's names): -Dhip-arch, -Dhipcc, -Dhip-include, -Dhip-device-lib.

const std = @import("std");

pub const Options = struct {
    arch: []const u8,
    hipcc: []const u8,
    include: []const u8,
    device_lib: []const u8,
    rocm: []const u8,
};

/// The exact architectures this port builds code objects for (module.zig machOf lists their ELF values).
const archs = [_][]const u8{"gfx1151"};

/// Device-code flags (ADAPTER.md 9.1): a raw code-object ELF (no offload bundle), wave32, no contraction, no fast math.
pub const hip_flags = [_][]const u8{
    "--offload-device-only", "--no-gpu-bundle-output", "-O3",                    "-std=c++20",
    "-fno-gpu-rdc",          "-mno-wavefrontsize64",   "-ffp-contract=off",      "-fno-fast-math",
    "-fdenormal-fp-math=ieee", "-D__HIP_PLATFORM_AMD__",
};

pub fn options(b: *std.Build) Options {
    // hipcc appends these after the explicit flags, so they would silently change the numerical contract (the guard
    // after the component kit, tools/dsv41-hip/build.sh there; tools/dsv41-hip/zig_build.sh here unsets them)
    for ([_][]const u8{ "HIPCC_COMPILE_FLAGS_APPEND", "HIPCC_LINK_FLAGS_APPEND" }) |name| {
        if (b.graph.environ_map.get(name)) |value| if (value.len != 0)
            std.debug.panic("{s} is set; the dsv41 HIP build's device flags are explicit: unset it (tools/dsv41-hip/zig_build.sh does)", .{name});
    }
    const arch = b.option([]const u8, "hip-arch", "Exact GPU architecture of the HIP code objects (default gfx1151)") orelse "gfx1151";
    var known = false;
    for (archs) |a| known = known or std.mem.eql(u8, a, arch);
    if (!known) std.debug.panic("-Dhip-arch={s}: this port builds for {s} only (no feature suffix, no neighbour gfx)", .{ arch, archs[0] });
    const hipcc = b.option([]const u8, "hipcc", "HIP compiler (default /usr/bin/hipcc)") orelse "/usr/bin/hipcc";
    const rocm = b.option([]const u8, "rocm", "ROCm root (default /opt/rocm)") orelse "/opt/rocm";
    const include = b.option([]const u8, "hip-include", "HIP header directory (default <rocm>/include)") orelse b.pathJoin(&.{ rocm, "include" });
    const device_lib = b.option([]const u8, "hip-device-lib", "ROCm device bitcode directory (default <rocm>/lib/llvm/amdgcn/bitcode)") orelse b.pathJoin(&.{ rocm, "lib", "llvm", "amdgcn", "bitcode" });
    return .{ .arch = arch, .hipcc = hipcc, .include = include, .device_lib = device_lib, .rocm = rocm };
}

/// The `gpu` module of a HIP build (zig/src/gpu/hip_facade.zig), with the two C files that read the installed headers.
pub fn gpuModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, o: Options) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/gpu/hip_facade.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("dsv41_abi", abiModule(b, target, optimize));
    m.addIncludePath(.{ .cwd_relative = o.include });
    const cflags = &[_][]const u8{ "-std=c11", "-D__HIP_PLATFORM_AMD__", "-Wall", "-Werror" };
    m.addCSourceFile(.{ .file = b.path("zig/src/gpu/hip/device_arch.c"), .flags = cflags });
    m.addCSourceFile(.{ .file = b.path("zig/src/gpu/hip/hip_abi_check.c"), .flags = cflags });
    return m;
}

/// zig/kernels/hip/dsv41/abi.h translated for Zig: TF_DS_HIP_ABI_VERSION and the by-value kernel structs, so each
/// struct has one definition for hipcc and Zig.
pub fn abiModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.addTranslateC(.{ .root_source_file = b.path("zig/kernels/hip/dsv41/abi.h"), .target = target, .optimize = optimize }).createModule();
}

/// tf-hip-kit (CPU only: no HIP runtime, no GPU).
pub fn kitTool(b: *std.Build) *std.Build.Step.Compile {
    const host = b.graph.host;
    return b.addExecutable(.{ .name = "tf-hip-kit", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/gpu/kit_tool.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{.{ .name = "dsv41_abi", .module = abiModule(b, host, .ReleaseSafe) }},
    }) });
}

/// hipcc --version's text, captured at make time and never cached, so a toolchain change rebuilds every code object
/// (zig/build/cuda.zig does the same for nvcc).
pub fn hipccVersion(b: *std.Build, o: Options) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ o.hipcc, "--version" });
    run.has_side_effects = true;
    return run.captureStdOut(.{});
}

/// How a kit is made beyond its sources: `declared` requires only what the sources declare (a partial kit for tests),
/// `registry` adds the registry's lists for the kit (the serve kit); `triton` imports w-tri's AOT Triton kit.
pub const KitMode = struct { required: enum { registry, declared } = .registry, triton: ?[]const u8 = null };

/// A kit directory (objects/ + manifest.json) for `kit` from (source name, code object name) pairs, by tf-hip-kit scan.
/// The manifest names its inputs by content (each source, abi.h, the registry, each Triton object) and the hipcc that
/// built them.
pub fn buildKit(b: *std.Build, o: Options, tool: *std.Build.Step.Compile, kit: []const u8, sources: []const [2][]const u8, mode: KitMode) std.Build.LazyPath {
    const version = hipccVersion(b, o);
    const run = b.addRunArtifact(tool);
    run.addArgs(&.{ "scan", "--kit", kit, "--arch", o.arch, "--required", @tagName(mode.required) });
    if (mode.triton) |dir| {
        run.addArg("--triton");
        run.addDirectoryArg(.{ .cwd_relative = dir });
        run.has_side_effects = true; // a directory outside the build: rescanned every time (its objects are hashed)
    }
    run.addArg("--registry");
    run.addFileArg(b.path("zig/kernels/hip/dsv41/registry.json"));
    run.addArg("--hipcc-version-file");
    run.addFileArg(version);
    for (hip_flags) |f| run.addArgs(&.{ "--flag", f });
    for ([_][]const u8{ "registry.json", "abi.h" }) |name| {
        run.addArg("--input");
        run.addPrefixedFileArg(b.fmt("{s}=", .{name}), b.path(b.fmt("zig/kernels/hip/dsv41/{s}", .{name})));
    }
    for (sources) |src| {
        run.addArg("--input");
        run.addPrefixedFileArg(b.fmt("{s}=", .{src[0]}), b.path(b.fmt("zig/kernels/hip/dsv41/{s}", .{src[0]})));
        run.addPrefixedFileArg(b.fmt("{s}=", .{src[0]}), codeObject(b, o, b.fmt("zig/kernels/hip/dsv41/{s}", .{src[0]}), src[1], version));
    }
    run.addArg("--out");
    return run.addOutputDirectoryArg(b.fmt("{s}-kit", .{kit}));
}

/// The `gpu` module of a CUDA build over the untouched CUDA runtime, host-only (no kernel images, no nvcc): the same
/// recipe as zig/build/cuda.zig's runtime() with no images.
pub fn cudaGpuModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const kernel_options = b.addOptions();
    kernel_options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", kernel_options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/gpu/cuda_facade.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    return m;
}

/// One .hip file as a raw code object for `o.arch`.
pub fn codeObject(b: *std.Build, o: Options, src: []const u8, name: []const u8, version: std.Build.LazyPath) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ o.hipcc, b.fmt("--offload-arch={s}", .{o.arch}) });
    run.addFileInput(version);
    run.addFileInput(b.path("zig/kernels/hip/dsv41/abi.h"));
    run.addArgs(&hip_flags);
    run.addArg(b.fmt("--rocm-path={s}", .{o.rocm}));
    run.addArg(b.fmt("--rocm-device-lib-path={s}", .{o.device_lib}));
    run.addPrefixedDirectoryArg("-I", b.path("zig/kernels/hip/dsv41"));
    run.addFileArg(b.path(src));
    run.addArg("-o");
    return run.addOutputFileArg(name);
}

/// A module embedding one code object: `arch` and `image` (8-byte aligned, as hipModuleLoadData wants).
fn embedded(b: *std.Build, o: Options, object: std.Build.LazyPath, file: []const u8) *std.Build.Module {
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(object, file);
    return b.createModule(.{ .root_source_file = wf.add("image.zig", b.fmt("pub const arch = \"{s}\";\npub const image align(8) = @embedFile(\"{s}\").*;\n", .{ o.arch, file })) });
}

/// The backend-neutral core (checkpoint formats, safetensors, the tokenizer) as the family imports it.
fn coreModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return coreWith(b, target, optimize, tokenizerModule(b, target, optimize));
}

/// The core over a given tokenizer module (a graph holds each file in one module: the server imports the tokenizer too).
fn coreWith(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, tokenizer: *std.Build.Module) *std.Build.Module {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    return core;
}

fn tokenizerModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

fn lanesModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

/// HIP family module: verbs loads only when selected; the HIP graph has no CUDA or NCCL module.
fn familyModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, o: Options, gpu: *std.Build.Module, lanes: *std.Build.Module, root: []const u8) *std.Build.Module {
    return familyWith(b, target, optimize, o, gpu, lanes, coreModule(b, target, optimize), root);
}

fn familyWith(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, o: Options, gpu: *std.Build.Module, lanes: *std.Build.Module, core: *std.Build.Module, root: []const u8) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("gpu", gpu);
    m.addImport("lanes", lanes);
    m.addImport("core", core);
    m.addImport("fabric", b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    m.addIncludePath(.{ .cwd_relative = o.include });
    m.addCSourceFile(.{ .file = b.path("zig/src/families/dsv41/hipblas_abi_check.c"), .flags = &.{ "-std=c11", "-D__HIP_PLATFORM_AMD__", "-Wall", "-Werror" } });
    // hip_load.zig's test holds its kernel specs to the registry
    m.addAnonymousImport("dsv41_registry", .{ .root_source_file = b.path("zig/kernels/hip/dsv41/registry.json") });
    return m;
}

/// The family's CPU side as module "dsv41" with no GPU module (zig/src/families/dsv41/cpu.zig).
fn cpuFamilyModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/cpu.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("core", coreModule(b, target, optimize));
    return m;
}

/// A check that `exe` names no NVIDIA library or driver entry point (they would be dlopen strings, which ldd misses).
fn noNvidia(b: *std.Build, exe: *std.Build.Step.Compile) *std.Build.Step {
    const run = b.addSystemCommand(&.{ "sh", "-c", "if strings -a \"$0\" | grep -E 'libcuda\\.so|libnccl|libcublas|libnvrtc|libcudart|cuInit|ncclCommInitRank|cublasCreate'; then echo \"$0 names an NVIDIA library\" >&2; exit 1; fi" });
    run.addArtifactArg(exe);
    run.has_side_effects = true;
    return &run.step;
}

/// llvm-objdump -d of a code object (tf-hip-kit isa-lint reads its mnemonics).
fn disassembly(b: *std.Build, o: Options, object: std.Build.LazyPath) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ b.pathJoin(&.{ o.rocm, "lib", "llvm", "bin", "llvm-objdump" }), "-d", "--no-show-raw-insn" });
    run.addFileArg(object);
    return run.captureStdOut(.{});
}

/// llvm-readelf --notes of a code object (the independent reader hsaco_xcheck compares against).
fn readelfNotes(b: *std.Build, o: Options, object: std.Build.LazyPath) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ b.pathJoin(&.{ o.rocm, "lib", "llvm", "bin", "llvm-readelf" }), "--notes" });
    run.addFileArg(object);
    return run.captureStdOut(.{});
}

/// The dsv41 kit's HIP sources (registry.json, kit "dsv41") and their code objects' names.
const native_sources = [_][2][]const u8{ .{ "glue.hip", "glue.hsaco" }, .{ "aux.hip", "aux.hsaco" }, .{ "aux_attn.hip", "aux_attn.hsaco" }, .{ "aux_index.hip", "aux_index.hsaco" }, .{ "aux_hc.hip", "aux_hc.hsaco" }, .{ "exl3.hip", "exl3.hsaco" }, .{ "exl3_pf.hip", "exl3_pf.hsaco" }, .{ "coll.hip", "coll.hsaco" }, .{ "aux_gate.hip", "aux_gate.hsaco" }, .{ "aux_rope.hip", "aux_rope.hsaco" }, .{ "dense.hip", "dense.hsaco" } };

/// The engine graph of a HIP server or its host tests: the shared tokenizer, core, lanes and engine API, the HIP `gpu`
/// module, the dsv41 family over them, and native/hip.zig (the engine host) as `native_engines`.
const Engines = struct { tokenizer: *std.Build.Module, api: *std.Build.Module, engines: *std.Build.Module, dsv41: *std.Build.Module, gpu: *std.Build.Module, lanes: *std.Build.Module };

fn hipEngines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, o: Options) Engines {
    const tokenizer = tokenizerModule(b, target, optimize);
    const core = coreWith(b, target, optimize, tokenizer);
    const lanes = lanesModule(b, target, optimize);
    const gpu = gpuModule(b, target, optimize, o);
    const dsv41 = familyWith(b, target, optimize, o, gpu, lanes, core, "zig/src/families/dsv41/dsv41.zig");
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const engines = b.createModule(.{
        .root_source_file = b.path("zig/src/native/hip.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "gpu", .module = gpu }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "dsv41", .module = dsv41 } },
    });
    return .{ .tokenizer = tokenizer, .api = api, .engines = engines, .dsv41 = dsv41, .gpu = gpu, .lanes = lanes };
}

/// Steps: test-dsv41-hip-host (no GPU work), dsv41-cpu (CPU tools), dsv41-hip-gpu-test (builds the GPU test programs
/// and the test kit; run them with tools/dsv41-hip/run_gpu_tests.sh, which takes the GPU lock).
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options) void {
    const o = options(b);
    const triton_kit = b.option([]const u8, "hip-triton-kit", "w-tri's AOT Triton kit (aot.json, hsaco/) folded into the installed dsv41 kit");
    const host = b.graph.host;
    const tool = kitTool(b);
    const cpu = b.step("dsv41-cpu", "dsv41 CPU tools with no GPU module in their graph: tf-hip-kit; tf-dsv41-check, tf-dsv41-rank-cache, tf-dsv41-engram-check into zig-out/dsv41-cpu/bin");
    cpu.dependOn(&b.addInstallArtifact(tool, .{}).step);
    const cpu_family = cpuFamilyModule(b, target, optimize);
    for ([_][2][]const u8{ .{ "tf-dsv41-check", "check_main.zig" }, .{ "tf-dsv41-rank-cache", "rank_cache_main.zig" }, .{ "tf-dsv41-engram-check", "engram_check_main.zig" } }) |t| {
        const exe = b.addExecutable(.{ .name = t[0], .root_module = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/src/families/dsv41/{s}", .{t[1]})),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "dsv41", .module = cpu_family }},
        }) });
        cpu.dependOn(&b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "dsv41-cpu/bin" } } }).step);
        cpu.dependOn(noNvidia(b, exe));
    }

    const host_test = b.step("test-dsv41-hip-host", "dsv41 HIP backend, gpu facades, kit and ABI tooling: host tests incl. a metadata cross-check against llvm-readelf (no GPU work)");
    host_test.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = gpuModule(b, host, .Debug, o) })).step);
    host_test.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cudaGpuModule(b, host, .Debug) })).step);
    const version = hipccVersion(b, o);
    const smoke_object = codeObject(b, o, "zig/kernels/hip/dsv41/smoke.hip", "smoke.hsaco", version);
    const echo_object = codeObject(b, o, "zig/kernels/hip/dsv41/abi_echo.hip", "abi_echo.hsaco", version);
    const xfiles = b.addWriteFiles();
    _ = xfiles.addCopyFile(smoke_object, "smoke.hsaco");
    _ = xfiles.addCopyFile(echo_object, "echo.hsaco");
    _ = xfiles.addCopyFile(readelfNotes(b, o, smoke_object), "smoke.txt");
    _ = xfiles.addCopyFile(readelfNotes(b, o, echo_object), "echo.txt");
    const xobjects = b.createModule(.{ .root_source_file = xfiles.add("objects.zig",
        \\pub const smoke align(8) = @embedFile("smoke.hsaco").*;
        \\pub const smoke_readelf = @embedFile("smoke.txt");
        \\pub const echo align(8) = @embedFile("echo.hsaco").*;
        \\pub const echo_readelf = @embedFile("echo.txt");
        \\
    ) });
    const xcheck = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_hip/hsaco_xcheck.zig"),
        .target = host,
        .link_libc = true,
        .imports = &.{ .{ .name = "gpu", .module = gpuModule(b, host, .Debug, o) }, .{ .name = "xcheck_objects", .module = xobjects } },
    }) });
    host_test.dependOn(&b.addRunArtifact(xcheck).step);
    host_test.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tool.root_module })).step);
    // the serve kit's native half: every registered glue/aux kernel compiled, none spilling, the manifest re-checked
    const native_kit = buildKit(b, o, tool, "dsv41", &native_sources, .{ .required = .declared });
    const verify = b.addRunArtifact(tool);
    verify.addArg("verify");
    verify.addDirectoryArg(native_kit);
    verify.addArgs(&.{ "--arch", o.arch });
    host_test.dependOn(&verify.step);
    const lint = b.addRunArtifact(tool);
    lint.addArg("lint");
    lint.addDirectoryArg(b.path("zig/kernels/hip"));
    lint.has_side_effects = true; // a new source file is not a tracked input of this step
    host_test.dependOn(&lint.step);
    // the native kernels' ISA: no fp16 conversion folded into a mixed-precision fma unless the registry allows it
    const isa_lint = b.addRunArtifact(tool);
    isa_lint.addArgs(&.{ "isa-lint", "--registry" });
    isa_lint.addFileArg(b.path("zig/kernels/hip/dsv41/registry.json"));
    const isa_version = hipccVersion(b, o);
    for (native_sources) |src| isa_lint.addPrefixedFileArg(b.fmt("{s}=", .{src[0]}), disassembly(b, o, codeObject(b, o, b.fmt("zig/kernels/hip/dsv41/{s}", .{src[0]}), src[1], isa_version)));
    host_test.dependOn(&isa_lint.step);
    // the family on HIP: its serving path analyzed from native.open, its files' host tests, the CPU root's tests
    const hip_gpu_host = gpuModule(b, host, .Debug, o);
    const hip_lanes_host = lanesModule(b, host, .Debug);
    host_test.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = familyModule(b, host, .Debug, o, hip_gpu_host, hip_lanes_host, "zig/src/families/dsv41/hip_test.zig") })).step);
    // the layer gate's own tests (its fixture reads, checks and dump around the replay cut; no GPU work)
    host_test.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/dsv41/layer_main.zig"),
        .target = host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "gpu", .module = hip_gpu_host }, .{ .name = "dsv41", .module = familyModule(b, host, .Debug, o, hip_gpu_host, hip_lanes_host, "zig/src/families/dsv41/dsv41.zig") } },
    }) })).step);
    host_test.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cpuFamilyModule(b, host, .Debug) })).step);
    // the HIP engine host's own tests (native/hip.zig over the whole family graph, as the server links it)
    host_test.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = hipEngines(b, host, .Debug, o).engines })).step);
    // rank 1's process (tf-dsv41-lanes) for HIP: built (not run), and free of NVIDIA library names
    const hip_gpu = gpuModule(b, target, optimize, o);
    const hip_lanes = lanesModule(b, target, optimize);
    const rank1 = b.addExecutable(.{ .name = "tf-dsv41-lanes", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/dsv41/lanes_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "gpu", .module = hip_gpu }, .{ .name = "dsv41", .module = familyModule(b, target, optimize, o, hip_gpu, hip_lanes, "zig/src/families/dsv41/dsv41.zig") }, .{ .name = "lanes", .module = hip_lanes } },
    }) });
    host_test.dependOn(&rank1.step);
    host_test.dependOn(noNvidia(b, rank1));

    const gpu_test = b.step("dsv41-hip-gpu-test", "Build tf-dsv41-hip-smoke, tf-dsv41-hip-abi-test, tf-dsv41-hip-glue-test, the test kit and the dsv41 native kit (GPU; run under the GPU lock with tools/dsv41-hip/run_gpu_tests.sh)");
    const smoke_kernels = embedded(b, o, smoke_object, "smoke.hsaco");
    const install_test = std.Build.InstallDir{ .custom = "native-hip/test" };
    const smoke = b.addExecutable(.{ .name = "tf-dsv41-hip-smoke", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_hip/smoke.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "gpu", .module = gpuModule(b, target, optimize, o) }, .{ .name = "smoke_kernels", .module = smoke_kernels } },
    }) });
    gpu_test.dependOn(&b.addInstallArtifact(smoke, .{ .dest_dir = .{ .override = install_test } }).step);
    const abi_test = b.addExecutable(.{ .name = "tf-dsv41-hip-abi-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_hip/abi_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "gpu", .module = gpuModule(b, target, optimize, o) }},
    }) });
    gpu_test.dependOn(&b.addInstallArtifact(abi_test, .{ .dest_dir = .{ .override = install_test } }).step);
    const test_kit = buildKit(b, o, tool, "test", &.{.{ "abi_echo.hip", "abi_echo.hsaco" }}, .{ .required = .declared });
    gpu_test.dependOn(&b.addInstallDirectory(.{ .source_dir = test_kit, .install_dir = install_test, .install_subdir = "kit" }).step);
    // the glue kernels and hipBLAS through the family's launchers, against host references (ReleaseSafe: the FP64
    // reference GEMMs take seconds, not minutes)
    gpu_test.dependOn(&b.addInstallDirectory(.{ .source_dir = native_kit, .install_dir = install_test, .install_subdir = "dsv41-kit" }).step);
    const rs_gpu = gpuModule(b, target, .ReleaseSafe, o);
    const rs_family = familyModule(b, target, .ReleaseSafe, o, rs_gpu, lanesModule(b, target, .ReleaseSafe), "zig/src/families/dsv41/dsv41.zig");
    const glue_test = b.addExecutable(.{ .name = "tf-dsv41-hip-glue-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_hip/glue_test.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "gpu", .module = rs_gpu }, .{ .name = "dsv41", .module = rs_family } },
    }) });
    gpu_test.dependOn(&b.addInstallArtifact(glue_test, .{ .dest_dir = .{ .override = install_test } }).step);

    // the serve kit: glue.hip and aux.hip (and w-exl3's sources as they land), w-tri's Triton kit with -Dhip-triton-kit,
    // every kernel the registry requires for dsv41 checked when a model opens it
    const serve_kit = buildKit(b, o, tool, "dsv41", &native_sources, .{ .triton = triton_kit });
    const kit_dir = b.fmt("native-hip/share/tensorfold/hip/{s}/dsv41", .{o.arch});
    const install_kit = b.addInstallDirectory(.{ .source_dir = serve_kit, .install_dir = .prefix, .install_subdir = kit_dir });
    b.step("dsv41-hip-kit", "The dsv41 AMD kit alone into zig-out/native-hip/share/tensorfold/hip/<arch>/dsv41 (-Dhip-triton-kit adds w-tri's Triton kit)").dependOn(&install_kit.step);

    // rank 0's server (tensorfold-native with the HIP engine host) and rank 1 (tf-dsv41-lanes), tf-hip-kit, the kit
    const native = b.step("native-hip", "tensorfold-native (rank 0) and tf-dsv41-lanes (rank 1) for HIP, tf-hip-kit and the dsv41 kit into zig-out/native-hip");
    const eng = hipEngines(b, target, optimize, o);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (as zig/build/cuda.zig's server)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const server = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = eng.api }, .{ .name = "tokenizer", .module = eng.tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = eng.engines }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = eng.engines }} }) } },
    }) });
    server.root_module.addOptions("build_options", build_options);
    const bin = std.Build.InstallDir{ .custom = "native-hip/bin" };
    const install_server = b.addInstallArtifact(server, .{ .dest_dir = .{ .override = bin } });
    const install_rank1 = b.addInstallArtifact(rank1, .{ .dest_dir = .{ .override = bin } });
    native.dependOn(&install_server.step);
    native.dependOn(&install_rank1.step);
    // the layer gate on HIP (tf-dsv41-layer: fixtures, --dump / --anchor / --replay-gather for w-ref's layer grading)
    const layer_gate = b.addExecutable(.{ .name = "tf-dsv41-layer", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/dsv41/layer_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "gpu", .module = eng.gpu }, .{ .name = "dsv41", .module = eng.dsv41 } },
    }) });
    native.dependOn(&b.addInstallArtifact(layer_gate, .{ .dest_dir = .{ .override = bin } }).step);
    native.dependOn(&b.addInstallArtifact(tool, .{ .dest_dir = .{ .override = bin } }).step);
    native.dependOn(&install_kit.step);
    // no NVIDIA library or entry point anywhere in the HIP binaries (their module graphs hold no CUDA or NCCL
    // module, so an import of one would not compile; this checks the linked bytes)
    const no_cuda = b.step("dsv41-hip-no-cuda", "Check the native-hip binaries for NVIDIA library and entry-point names");
    no_cuda.dependOn(noNvidia(b, server));
    no_cuda.dependOn(noNvidia(b, rank1));
    no_cuda.dependOn(noNvidia(b, layer_gate));
    native.dependOn(no_cuda);
    // w-link: the tcp_host collectives between the two boxes over the USB4 link (both ranks; locks on both boxes)
    const coll_check = b.addExecutable(.{ .name = "tf-dsv41-coll-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/dsv41_hip/coll_check.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "gpu", .module = rs_gpu }, .{ .name = "dsv41", .module = rs_family }, .{ .name = "smoke_kernels", .module = smoke_kernels } },
    }) });
    gpu_test.dependOn(&b.addInstallArtifact(coll_check, .{ .dest_dir = .{ .override = install_test } }).step);
}
