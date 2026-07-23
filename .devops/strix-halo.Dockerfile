# Publishable Toolbx image: ROCm + THIS repo's llama.cpp for AMD Strix Halo (gfx1151).
# Build from the repo root (build context = this checkout, source is COPYed in):
#   podman build --no-cache -t llama-rocm-strixhalo -f .devops/strix-halo.Dockerfile .
# Toolbx:
#   toolbox create llama-rocm-strixhalo -i localhost/llama-rocm-strixhalo -- \
#     --device /dev/dri --device /dev/kfd --group-add video --group-add render \
#     --group-add sudo --security-opt seccomp=unconfined

# ---------------------------------------------------------------------------
# build stage
# ---------------------------------------------------------------------------
FROM registry.fedoraproject.org/fedora:43 AS builder

# ROCm repo. Override --build-arg ROCM_BASEURL=... to switch ROCm version.
ARG ROCM_BASEURL=https://repo.radeon.com/rocm/rhel10/7.2.4/main
RUN <<EOF
tee /etc/yum.repos.d/rocm.repo <<REPO
[ROCm]
name=ROCm
baseurl=${ROCM_BASEURL}
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO
EOF

RUN dnf -y --nodocs --setopt=install_weak_deps=False \
      --exclude='*sdk*' --exclude='*samples*' --exclude='*-doc*' --exclude='*-docs*' \
      install \
        make gcc gcc-c++ cmake lld clang clang-devel compiler-rt libcurl-devel ninja-build ccache \
        rocm-llvm rocm-device-libs hip-runtime-amd hip-devel \
        rocblas rocblas-devel hipblas hipblas-devel rocm-cmake libomp-devel libomp \
        rocminfo radeontop git-core \
    && dnf clean all && rm -rf /var/cache/dnf/*

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode \
    HIP_PLATFORM=amd \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

# ---- build THIS repo (source copied from the build context) ------------------
WORKDIR /opt/llama.cpp
COPY . .

# gfx1151 = Strix Halo. UMA flags on for unified memory.
RUN cmake -S . -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1151 \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_RPC=ON \
        -DLLAMA_HIP_UMA=ON \
        -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON \
        -DLLAMA_CURL=ON \
        -DROCM_PATH=/opt/rocm \
        -DHIP_PATH=/opt/rocm \
        -DHIP_PLATFORM=amd \
    && cmake --build build --config Release -- -j$(nproc) \
    && cmake --install build --config Release

RUN mkdir -p /usr/local/lib64 \
    && find /opt/llama.cpp/build -type f -name 'lib*.so*' -exec cp {} /usr/local/lib64/ \; \
    && ldconfig

# ---------------------------------------------------------------------------
# runtime stage
# ---------------------------------------------------------------------------
FROM registry.fedoraproject.org/fedora-minimal:43

ARG ROCM_BASEURL=https://repo.radeon.com/rocm/rhel10/7.2.4/main
RUN <<EOF
tee /etc/yum.repos.d/rocm.repo <<REPO
[ROCm]
name=ROCm
baseurl=${ROCM_BASEURL}
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO
EOF

RUN microdnf -y --nodocs --setopt=install_weak_deps=0 \
      --exclude='*sdk*' --exclude='*samples*' --exclude='*-doc*' --exclude='*-docs*' \
      install \
        bash ca-certificates libatomic libstdc++ libgcc libgomp libcurl sudo \
        hip-runtime-amd rocblas hipblas \
        rocminfo radeontop procps-ng \
    && microdnf clean all && rm -rf /var/cache/dnf/*

COPY --from=builder /usr/local/ /usr/local/

RUN echo "/usr/local/lib"    >  /etc/ld.so.conf.d/local.conf \
    && echo "/usr/local/lib64" >> /etc/ld.so.conf.d/local.conf \
    && ldconfig

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    PATH=/opt/rocm/bin:/usr/local/bin:$PATH

# Marks the image as a Toolbx-compatible container.
LABEL com.github.containers.toolbox="true"

CMD ["/bin/bash"]
