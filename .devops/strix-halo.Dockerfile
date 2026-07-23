# Publishable Toolbx image: ROCm 7 nightly (TheRock, gfx1151) + THIS repo's llama.cpp
# for AMD Strix Halo. Build from the repo root (source is COPYed in, no clone):
#   podman build --no-cache -t llama-rocm-strixhalo -f .devops/strix-halo.Dockerfile .
# Toolbx:
#   toolbox create llama-rocm-strixhalo -i localhost/llama-rocm-strixhalo -- \
#     --device /dev/dri --device /dev/kfd --group-add video --group-add render \
#     --group-add sudo --security-opt seccomp=unconfined
#
# ROCm comes from the TheRock nightly tarballs (the "7.14" alpha line), not a dnf
# repo. Pin a specific build with --build-arg ROCM_TARBALL=<key> if needed.

# ---------------------------------------------------------------------------
# build stage
# ---------------------------------------------------------------------------
FROM registry.fedoraproject.org/fedora:43 AS builder

RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
      make gcc gcc-c++ cmake lld clang clang-devel compiler-rt libcurl-devel \
      git patch curl ninja-build tar xz aria2 ccache \
    && dnf clean all && rm -rf /var/cache/dnf/*

# Fetch the latest Linux ROCm 7.x tarball for gfx1151 (TheRock nightly), or a
# pinned one via ROCM_TARBALL. Extract into /opt/rocm-7.0.
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
ARG ROCM_TARBALL=
RUN set -euo pipefail; \
    BASE="https://therock-nightly-tarball.s3.amazonaws.com"; \
    PREFIX="therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}"; \
    KEY="${ROCM_TARBALL}"; \
    if [ -z "${KEY}" ]; then \
      KEY="$(curl -s "${BASE}?list-type=2&prefix=${PREFIX}" \
        | tr '<' '\n' \
        | grep -o "therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}\..*\.tar\.gz" \
        | sort -V | tail -n1)"; \
    fi; \
    echo "ROCm tarball: ${KEY}"; \
    aria2c -x 16 -s 16 -j 16 --file-allocation=none "${BASE}/${KEY}" -o therock.tar.gz
RUN mkdir -p /opt/rocm-7.0 \
    && tar xzf therock.tar.gz -C /opt/rocm-7.0 --strip-components=1 \
    && rm -f therock.tar.gz

ENV ROCM_PATH=/opt/rocm-7.0 \
    HIP_PLATFORM=amd \
    HIP_PATH=/opt/rocm-7.0 \
    HIP_CLANG_PATH=/opt/rocm-7.0/llvm/bin \
    HIP_INCLUDE_PATH=/opt/rocm-7.0/include \
    HIP_LIB_PATH=/opt/rocm-7.0/lib \
    HIP_DEVICE_LIB_PATH=/opt/rocm-7.0/lib/llvm/amdgcn/bitcode \
    PATH=/opt/rocm-7.0/bin:/opt/rocm-7.0/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/rocm-7.0/lib:/opt/rocm-7.0/lib64:/opt/rocm-7.0/llvm/lib \
    LIBRARY_PATH=/opt/rocm-7.0/lib:/opt/rocm-7.0/lib64 \
    CPATH=/opt/rocm-7.0/include \
    PKG_CONFIG_PATH=/opt/rocm-7.0/lib/pkgconfig

# ---- build THIS repo (source copied from the build context) ------------------
WORKDIR /opt/llama.cpp
COPY . .

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

# Trim headers/docs/static libs from the ROCm tree to shrink the runtime copy.
RUN find /opt/rocm-7.0 -type f -name '*.a' -delete \
    && rm -rf /opt/rocm-7.0/include /opt/rocm-7.0/share \
              /opt/rocm-7.0/llvm/include /opt/rocm-7.0/llvm/share

# ---------------------------------------------------------------------------
# runtime stage
# ---------------------------------------------------------------------------
FROM registry.fedoraproject.org/fedora-minimal:43

RUN microdnf -y --nodocs --setopt=install_weak_deps=0 install \
      bash ca-certificates libatomic libstdc++ libgcc libgomp libcurl \
      radeontop procps-ng sudo \
    && microdnf clean all && rm -rf /var/cache/dnf/*

COPY --from=builder /opt/rocm-7.0 /opt/rocm-7.0
COPY --from=builder /usr/local/ /usr/local/

ENV ROCM_PATH=/opt/rocm-7.0 \
    HIP_PLATFORM=amd \
    HIP_PATH=/opt/rocm-7.0 \
    HIP_CLANG_PATH=/opt/rocm-7.0/llvm/bin \
    HIP_DEVICE_LIB_PATH=/opt/rocm-7.0/lib/llvm/amdgcn/bitcode \
    PATH=/opt/rocm-7.0/bin:/opt/rocm-7.0/llvm/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/rocm-7.0/lib:/opt/rocm-7.0/lib64:/opt/rocm-7.0/llvm/lib

RUN echo "/usr/local/lib"    >  /etc/ld.so.conf.d/local.conf \
    && echo "/usr/local/lib64" >> /etc/ld.so.conf.d/local.conf \
    && ldconfig

# Marks the image as a Toolbx-compatible container.
LABEL com.github.containers.toolbox="true"

CMD ["/bin/bash"]
