# Strix Halo llama.cpp Toolbox

A Toolbx/Distrobox-ready ROCm container that ships **this repository's** optimized
llama.cpp build for AMD Ryzen AI Max "Strix Halo" APUs (RDNA 3.5 iGPU, `gfx1151`).

Unlike a generic image, this one is compiled from the local source tree, so it
carries the Strix-Halo-specific kernel work assembled in this fork:

- **Quantized-KV Flash Attention (TILE path)** - this is **Nathan Wilson's**
  work, incorporated from
  [`strix-halo-fa-fixes`](https://github.com/Nathanw1014/llama.cpp/tree/strix-halo-fa-fixes).
  It is not original to this fork; see [Acknowledgements](#acknowledgements).
- RDNA 3.5 MMVQ two-wave tuning, MoE `mul_mat_id` specialization and gate/up MMQ
  pairing, tiled transposed concat, and adaptive Gated Delta Net - contributed
  in this fork.

## Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Running Inference](#running-inference)
- [Host Configuration](#host-configuration)
- [Building Locally](#building-locally)
- [Publishing](#publishing)
- [Files](#files)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Requirements

- AMD Ryzen AI Max "Strix Halo" (`gfx1151`).
- A recent kernel with stable `gfx1151` support (6.18.4+ recommended; see
  [Host Configuration](#host-configuration)).
- `podman` and `toolbox` (Fedora) or `distrobox` (Ubuntu/Debian/others).

## Quick Start

```sh
# Fedora (toolbox). Ubuntu/Debian users: use `distrobox` instead of `toolbox`.
toolbox create llama-rocm-strixhalo \
  --image docker.io/higaetan/strix-halo-llamacpp-toolbox:rocm-7.14 \
  -- --device /dev/dri --device /dev/kfd --group-add video --group-add render \
     --group-add sudo --security-opt seccomp=unconfined

toolbox enter llama-rocm-strixhalo
llama-cli --list-devices
```

Or use the helper, which pulls the latest image and (re)creates the container:

```sh
./toolbox/refresh-toolbox.sh
```

## Running Inference

> IMPORTANT: on Strix Halo always enable flash attention (`-fa 1`). For models
> that take a large fraction of RAM, add `--direct-io` (or `--no-mmap`). The
> default mmap path keeps the GGUF file resident AND allocates the same bytes
> again as GTT, so a ~69 GiB model needs ~138 GiB and thrashes unified memory.
> `--direct-io` streams weights through pinned buffers and loads in seconds.

Server (OpenAI-compatible API):

```sh
llama-server -m model.gguf -c 8192 -ngl 999 -fa 1 --direct-io
```

CLI:

```sh
llama-cli -m model.gguf -ngl 999 -fa 1 --direct-io -p "Write a Strix Halo haiku."
```

## Host Configuration

Add these kernel parameters to enable unified memory while reserving ~4 GiB for
the OS (max 124 GiB for the iGPU):

```
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856
```

| Parameter | Purpose |
| --- | --- |
| `amd_iommu=off` | Disables the AMD IOMMU; faster and more stable than `iommu=pt`. |
| `amdgpu.gttsize=126976` | Caps GPU unified memory at 124 GiB (126976 MiB / 1024). |
| `ttm.pages_limit=32505856` | Caps pinned memory at 124 GiB (32505856 x 4 KiB). |

Apply and reboot:

```sh
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

## Building Locally

The Dockerfile builds the current checkout (build context = repo root, source is
copied in - no clone), so the image always matches your working tree.

```sh
# from the repo root
podman build --no-cache -t llama-rocm-strixhalo -f .devops/strix-halo.Dockerfile .

toolbox create llama-rocm-strixhalo -i localhost/llama-rocm-strixhalo -- \
  --device /dev/dri --device /dev/kfd --group-add video --group-add render \
  --group-add sudo --security-opt seccomp=unconfined
```

ROCm 7.14 is taken from the proven local toolchain image (the same one the
working `rocm-7.14` toolbox is built on), defaulting to
`localhost/llama-rocm-7.14-build:7.14.0a20260612`. This is deliberate: the
matching TheRock nightly on `repo.amd.com/rocm/tarball/` gets pruned over time,
and the generic S3 nightly ships an HSA runtime that segfaults in
`GpuAgent::InitDma()` on gfx1151. Point at a different toolchain image with:

```sh
podman build --no-cache -t llama-rocm-strixhalo -f .devops/strix-halo.Dockerfile . \
  --build-arg ROCM_BUILD_IMAGE=localhost/llama-rocm-7.14-build:7.14.0a20260612
```

> The toolchain image must contain a working ROCm 7.14 at `/opt/rocm` plus
> `cmake`/`hipcc`. If you do not have it, build it from the TheRock gfx1151
> tarball (see `repo.amd.com/rocm/tarball/therock-dist-linux-gfx1151-<ver>`)
> while that version is still published.

## Publishing

Build the image locally (see above), then push it to Docker Hub by hand:

```sh
# log in once (create an access token: Docker Hub -> Account Settings ->
# Security -> New Access Token, with Read/Write scope)
podman login docker.io -u higaetan

# tag and push
podman tag llama-rocm-strixhalo docker.io/higaetan/strix-halo-llamacpp-toolbox:rocm-7.14
podman push docker.io/higaetan/strix-halo-llamacpp-toolbox:rocm-7.14
```

The repository is created automatically on first push; set it public under
Docker Hub -> the repo -> Settings -> Make public.

## Files

| Path | Role |
| --- | --- |
| `.devops/strix-halo.Dockerfile` | Multi-stage build: compile this repo, then a slim ROCm runtime. |
| `toolbox/refresh-toolbox.sh` | Pull the published image and (re)create the local toolbox. |
| `toolbox/README.md` | This document. |

## Troubleshooting

- **Toolbox cannot see the GPU** - confirm `/dev/dri` and `/dev/kfd` are passed
  and your user is in the `video` and `render` groups.
- **Large model hangs on load / RAM spikes** - you are on the mmap path; add
  `--direct-io`. See [Running Inference](#running-inference).
- **Build runs out of memory** - HIP compilation is heavy; give the builder more
  RAM/swap or lower the parallel job count.
- **`podman push` denied on Docker Hub** - run `podman login docker.io` with an
  access token that has Read/Write scope.

## Acknowledgements

- **[Nathanw1014/llama.cpp `strix-halo-fa-fixes`](https://github.com/Nathanw1014/llama.cpp/tree/strix-halo-fa-fixes)**
  (Nathan Wilson) - the quantized-KV Flash Attention TILE path in this fork **is
  his code**, not a reimplementation. It was taken near-verbatim from his commit
  `2a24abc63` ("CUDA: dequantize KV on load in the tile FA kernel, use it for
  quantized decode"): dequantizing the KV cache on load inside the tile FA kernel
  for quantized decode, plus the head-128/256 and gqa-ratio-8 quantized-KV tests
  and the Strix Halo benchmarks.

  To attribute this properly in the git history, his original code and its
  explanatory comments were restored in a dedicated commit **authored in his
  name** (`ggml-cuda: restore original comments for quantized-KV FA tile`,
  author: Nathan Wilson). The only edit this fork makes to that path is a
  one-line `if` -> `if constexpr` guard needed for the D64 `J=0` instantiation;
  everything else in the TILE dequant path is his. Full credit and authorship for
  it belong to him.
- **[kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)**
  - the packaging approach here (multi-stage Fedora + ROCm Dockerfile, refresh
  script, and the host kernel parameters) is modeled on that project.
  Huge thanks to kyuz0 for the reference work and benchmarks.
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** - the upstream
  project this fork is based on.
- **AMD ROCm** and the [TheRock](https://github.com/ROCm/TheRock) builds that make
  `gfx1151` usable.
- The Strix Halo community, including
  [deseven's home lab notes](https://strixhalo-homelab.d7.wtf/) and
  [lhl's testing builds](https://github.com/lhl/strix-halo-testing), for the
  hardware/driver groundwork.

## License

The container packaging in this directory follows the license of this
repository (MIT, inherited from llama.cpp). ROCm and Fedora components retain
their own respective licenses.
