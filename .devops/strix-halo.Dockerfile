# Publishable Toolbx/Distrobox image: ROCm 7.14 (gfx1151) + THIS repo's llama.cpp
# for AMD Strix Halo. Self-contained so it builds anywhere (incl. GitHub cloud
# runners): ROCm is downloaded from AMD's official multi-arch tarball mirror.
#
# Build from the repo root (source is COPYed in, no clone):
#   podman build --no-cache -t llama-rocm-strixhalo -f .devops/strix-halo.Dockerfile .
# Pin a different ROCm release:
#   --build-arg ROCM_VERSION=7.14.0
#
# NOTE: use the official repo.amd.com tarball (good HSA runtime), NOT the generic
# TheRock S3 nightly, whose HSA runtime segfaults in GpuAgent::InitDma() on gfx1151.

# ---------------------------------------------------------------------------
# build stage
# ---------------------------------------------------------------------------
FROM registry.fedoraproject.org/fedora:43 AS builder

RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
      make gcc gcc-c++ cmake lld clang clang-devel compiler-rt libcurl-devel \
      git patch curl ninja-build tar xz aria2 ccache \
    && dnf clean all && rm -rf /var/cache/dnf/*

# Official AMD ROCm release tarball for gfx1151 (Strix Halo).
# List: https://repo.amd.com/rocm/tarball-multi-arch/
ARG ROCM_VERSION=7.14.0
ARG GFX=gfx1151
WORKDIR /tmp
RUN set -eux; \
    URL="https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-${GFX}-${ROCM_VERSION}.tar.gz"; \
    aria2c -x 16 -s 16 -j 16 --file-allocation=none "$URL" -o rocm.tar.gz; \
    mkdir -p /opt/rocm; \
    tar -xzf rocm.tar.gz -C /opt/rocm; \
    rm -f rocm.tar.gz

ENV ROCM_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    HIP_PATH=/opt/rocm \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib

WORKDIR /opt/llama.cpp
COPY . .

RUN cmake -S . -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1151 \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_RPC=ON \
        -DLLAMA_HIP_UMA=ON \
    && cmake --build build --config Release -- -j"$(nproc)" \
    && cmake --install build --config Release

RUN mkdir -p /usr/local/lib64 \
    && find /opt/llama.cpp/build -type f -name 'lib*.so*' -exec cp {} /usr/local/lib64/ \; \
    && ldconfig

# Slim the ROCm tree to what llama.cpp loads at runtime (16 GB -> ~1.8 GB): drop
# compilers/headers/static libs, docs/tests, and unused libraries (MIOpen, RCCL,
# rocFFT/rocSPARSE/rocRAND, MLIR, rocjitsu, rocprof-sys). Keep hip/hsa,
# rocblas/hipblas/hipblaslt/rocsolver/rocroller, amd_comgr + libLLVM/libclang-cpp,
# the gfx1151 Tensile kernels, and the bundled rocm_sysdeps.
RUN cd /opt/rocm \
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

COPY --from=builder /opt/rocm /opt/rocm
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

LABEL com.github.containers.toolbox="true"

CMD ["/bin/bash"]
