#!/bin/sh
set -eu
export HIP_PLATFORM=amd
export HIP_PATH=/nix/store/al4qwi7jfyrdkyb25xsri8b00b1agj1r-clr-10.1
export HIP_CLANG_PATH=/nix/store/vrf2j5y8a7adjrg894rnpfn5m9w2pq7c-clang-24.0.0-rocm/bin
export PATH=$HIP_CLANG_PATH:/nix/store/40b0f6fagcxizyl7ljhib7975mbxjqfw-lld-24.0.0-rocm/bin:/nix/store/gzknyraaciq5ilgi09y2r0ig1pdfbrc3-llvm-24.0.0-rocm/bin:/root/.nix-profile/bin:/usr/bin:/bin
exec /nix/store/77hssc153waqmsd9yr8dfy39cs9c6qgm-hipcc-24.0.0-rocm/bin/hipcc \
  --gcc-toolchain=/nix/store/l3msfz6rylwsyhnjzsbr8ylfshh1xw5f-gcc-prefix \
  -O2 -std=c++17 -fuse-ld=lld --offload-arch=gfx1151 --hip-path="$HIP_PATH" \
  --hip-device-lib-path=/nix/store/gyzwfffwdwj8dpkrasiqd5hkr2adx844-rocm-device-libs-24.0.0-rocm/amdgcn/bitcode \
  -I/bench/include -I/nix/store/8r06yrrk214k5hhfwvwc8r66dsc19ajd-hip-common-10.1/include \
  /bench/tbv_gpu_verbs.hip -L/bench -libverbs -o /bench/tbv_gpu_verbs-current
