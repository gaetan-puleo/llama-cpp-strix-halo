# Publishable Toolbx image: ROCm 7.14 (gfx1151) + THIS repo's llama.cpp for AMD Strix Halo.
#
# ROCm comes from the proven local toolchain image (the same one the working
# `rocm-7.14` toolbox is built on). The matching TheRock nightly on the web
# (repo.amd.com/rocm/tarball/therock-dist-linux-gfx1151-<ver>) is pruned over
# time, and the generic S3 nightly ships an HSA runtime that segfaults in
# GpuAgent::InitDma() on gfx1151 -- so we reuse the known-good tree instead.
#
# Build from the repo root (source is COPYed in, no clone):
#   podman build --no-cache -t llama-rocm-strixhalo -f .devops/strix-halo.Dockerfile .
# Override the toolchain image if yours differs:
#   --build-arg ROCM_BUILD_IMAGE=localhost/llama-rocm-7.14-build:7.14.0a20260612

# ---------------------------------------------------------------------------
# build stage: compile against the proven ROCm 7.14 toolchain image
# ---------------------------------------------------------------------------
ARG ROCM_BUILD_IMAGE=localhost/llama-rocm-7.14-build:7.14.0a20260612
FROM ${ROCM_BUILD_IMAGE} AS builder

WORKDIR /opt/llama.cpp
COPY . .

# gfx1151 = Strix Halo. UMA on for unified memory. The toolchain image already
# sets ROCM_PATH=/opt/rocm (-> the working 7.14 tree) and provides hipcc.
RUN cmake -S . -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1151 \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_RPC=ON \
        -DLLAMA_HIP_UMA=ON \
    && cmake --build build --config Release -- -j$(nproc) \
    && cmake --install build --config Release

RUN mkdir -p /usr/local/lib64 \
    && find /opt/llama.cpp/build -type f -name 'lib*.so*' -exec cp {} /usr/local/lib64/ \; \
    && ldconfig

# Resolve the real ROCm dir (the image exposes it via the /opt/rocm symlink) and
# slim it to what llama.cpp actually loads at runtime: hip/hsa, rocblas/hipblas/
# hipblaslt/rocsolver/rocroller, amd_comgr + libLLVM/libclang-cpp, and the bundled
# rocm_sysdeps. Drop compilers (bin), static libs, docs/tests, and unused
# libraries (MIOpen, RCCL, rocFFT/rocSPARSE/rocRAND). Keeps a single ~4 GB layer.
RUN cp -a "$(readlink -f /opt/rocm)" /opt/rocm-dist \
    && cd /opt/rocm-dist \
    && rm -rf bin include share clients tests libhipcxx libexec lib/rdc lib/host-math \
    && rm -rf lib/llvm/bin lib/llvm/include lib/llvm/share \
    && find . -name '*.a' -delete \
    && rm -f lib/libMIOpen*.so* lib/librccl*.so* lib/librocfft*.so* \
             lib/librocsparse*.so* lib/librocrand*.so* lib/librocalution*.so* \
             lib/librocprof-sys*.so* lib/librocjitsu*.so* \
             lib/llvm/lib/libMLIR*.so* lib/llvm/lib/libclang.so*

# ---------------------------------------------------------------------------
# runtime stage
# ---------------------------------------------------------------------------
FROM registry.fedoraproject.org/fedora-minimal:43

RUN microdnf -y --nodocs --setopt=install_weak_deps=0 install \
      bash ca-certificates libatomic libstdc++ libgcc libgomp libcurl \
      numactl-libs libdrm elfutils-libelf pciutils-libs radeontop procps-ng sudo \
    && microdnf clean all && rm -rf /var/cache/dnf/*

# The proven ROCm 7.14 tree (working HSA runtime) + the freshly built llama.cpp.
COPY --from=builder /opt/rocm-dist /opt/rocm
COPY --from=builder /usr/local/ /usr/local/

RUN echo "/usr/local/lib"    >  /etc/ld.so.conf.d/local.conf \
    && echo "/usr/local/lib64" >> /etc/ld.so.conf.d/local.conf \
    && echo "/opt/rocm/lib"    >> /etc/ld.so.conf.d/local.conf \
    && echo "/opt/rocm/lib64"  >> /etc/ld.so.conf.d/local.conf \
    && ldconfig

ENV ROCM_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    HIP_PATH=/opt/rocm \
    HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib

# Marks the image as a Toolbx-compatible container.
LABEL com.github.containers.toolbox="true"

CMD ["/bin/bash"]
