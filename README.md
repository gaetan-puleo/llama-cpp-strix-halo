# llama.cpp for AMD Strix Halo (gfx1151)

An optimized [llama.cpp](https://github.com/ggml-org/llama.cpp) fork for the
**AMD Ryzen AI Max "Strix Halo"** APU (RDNA 3.5 iGPU, `gfx1151`, unified memory),
shipped as a ready-to-run **ROCm toolbox container**.

- **Image:** `docker.io/higaetan/strix-halo-llamacpp-toolbox:rocm-7.14`
- **Runtime:** Fedora + ROCm 7.14, ~1.9 GB (slimmed to what llama.cpp actually loads)
- **Works with:** `toolbox` (Fedora) and `distrobox` (Ubuntu/Debian/others)

## What's in this fork

On top of upstream llama.cpp, this fork carries Strix-Halo-specific CUDA/HIP
kernel work for RDNA 3.5:

- **Quantized-KV Flash Attention (TILE path)** - dequantize the KV cache once
  into SRAM on load and reuse it across every Q head that shares a KV head, so
  dequant work stops scaling with `gqa_ratio`. **This is Nathan Wilson's work**
  (see [Acknowledgements](#acknowledgements)); it makes quantized-KV decode fast.
- RDNA 3.5 MMVQ two-wave tuning
- MoE `mul_mat_id` specialization and gate/up MMQ pairing
- Tiled transposed concat
- Adaptive Gated Delta Net

Validated on Strix Halo (Laguna-S 2.1 Q4_K_XL, 69 GiB, full GPU offload):
prompt ~81 t/s, generation ~25 t/s.

## Quick start

Pull the published image and create the container. Works with both `toolbox`
and `distrobox` - note the different way each passes device flags.

**toolbox (Fedora):**

```sh
toolbox create strix-halo \
  --image docker.io/higaetan/strix-halo-llamacpp-toolbox:rocm-7.14 \
  -- --device /dev/dri --device /dev/kfd --group-add video --group-add render \
     --group-add sudo --security-opt seccomp=unconfined

toolbox enter strix-halo
```

**distrobox (Ubuntu/Debian/others):**

```sh
distrobox create --name strix-halo \
  --image docker.io/higaetan/strix-halo-llamacpp-toolbox:rocm-7.14 \
  --additional-flags "--device /dev/dri --device /dev/kfd --group-add video --group-add render --group-add sudo --security-opt seccomp=unconfined"

distrobox enter strix-halo
```

Then, inside the container:

```sh
llama-cli --list-devices
# ROCm0: AMD Radeon 8060S Graphics (126976 MiB, ...)
```

## Running inference

> IMPORTANT: on Strix Halo always enable flash attention (`-fa 1`) and disable
> mmap (`--no-mmap`). The default mmap path keeps the GGUF file resident AND
> allocates the same bytes again as GTT, so a ~69 GiB model needs ~138 GiB and
> thrashes unified memory. `--no-mmap` reads the weights straight into GTT.

```sh
# server (OpenAI-compatible API)
llama-server -m model.gguf -c 8192 -ngl 999 -fa 1 --no-mmap

# CLI
llama-cli -m model.gguf -ngl 999 -fa 1 --no-mmap -p "The capital of France is"
```

## Host configuration

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
sudo grub2-mkconfig -o /boot/grub2/grub.cfg && sudo reboot
```

## Building the image locally

The Dockerfile compiles this repo (source is copied in, no clone) and downloads
ROCm 7.14.0 from AMD's official multi-arch tarball mirror, so it builds anywhere
(including GitHub-hosted runners).

```sh
# from the repo root
podman build -t llama-rocm-strixhalo -f .devops/strix-halo.Dockerfile .

# pin a different ROCm release from repo.amd.com/rocm/tarball-multi-arch/
podman build -t llama-rocm-strixhalo -f .devops/strix-halo.Dockerfile . \
  --build-arg ROCM_VERSION=7.14.0
```

> Use the official `repo.amd.com` tarballs. The generic TheRock S3 nightly ships
> an HSA runtime that segfaults in `GpuAgent::InitDma()` on gfx1151.

## Publishing (CI)

`.github/workflows/publish-toolbox.yml` builds on a GitHub-hosted `ubuntu-latest`
runner and pushes to Docker Hub. It runs **manually** (Actions -> Run workflow)
and **nightly** (02:00 UTC). Configure two repo secrets once:

- `DOCKERHUB_USERNAME` = `higaetan`
- `DOCKERHUB_TOKEN` = a Docker Hub access token (Read/Write)

It publishes `rocm-7.14` (rolling) and `rocm-7.14_<date>` (immutable).

## Layout

| Path | Role |
| --- | --- |
| `.devops/strix-halo.Dockerfile` | Multi-stage build: compile this repo, then a slim ROCm runtime. |
| `.github/workflows/publish-toolbox.yml` | CI: build on `ubuntu-latest`, push to Docker Hub. |
| `toolbox/refresh-toolbox.sh` | Pull the published image and (re)create the local container. |
| `toolbox/README.md` | Toolbox-focused documentation. |

## Troubleshooting

- **Container cannot see the GPU** - confirm `/dev/dri` and `/dev/kfd` are passed
  and your user is in the `video` and `render` groups.
- **Large model hangs on load / RAM spikes** - you are on the mmap path; add
  `--no-mmap`.
- **CI build fails with "no space left"** - GitHub runners have limited disk; the
  workflow already frees space, but the extracted ROCm tree is large.

## Acknowledgements

- **[Nathanw1014/llama.cpp `strix-halo-fa-fixes`](https://github.com/Nathanw1014/llama.cpp/tree/strix-halo-fa-fixes)**
  (Nathan Wilson) - the quantized-KV Flash Attention TILE path in this fork **is
  his code**, taken near-verbatim from his commit `2a24abc63`. It is recorded in
  the git history with his authorship intact. Full credit and thanks to him.
- **[kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)**
  - the packaging approach (multi-stage Fedora + ROCm image, refresh script, CI,
  host kernel parameters) is modeled on that project.
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** - the upstream
  project this fork is based on.
- **AMD ROCm** and the [TheRock](https://github.com/ROCm/TheRock) builds that make
  `gfx1151` usable.
- The Strix Halo community, including
  [deseven's home lab](https://strixhalo-homelab.d7.wtf/) and
  [lhl's testing builds](https://github.com/lhl/strix-halo-testing).

## License

MIT, inherited from llama.cpp. See [LICENSE](LICENSE). ROCm and Fedora components
retain their own respective licenses.
