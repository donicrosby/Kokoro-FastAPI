# Kokoro-FastAPI ROCm gfx1151 (Strix Halo) Container

This directory contains the Docker build for AMD Strix Halo iGPUs (gfx1151 / RDNA 3.5) using ROCm 7.2.4.

> **Note:** This is a separate image from `docker/rocm/` because ROCm 7.2.4 drops support for older GPU architectures (gfx900, gfx906) that are only supported through ROCm 6.4.4. If you have an older AMD GPU, use `docker/rocm/` instead.

## Requirements

- AMD Ryzen AI MAX+ 395 or similar Strix Halo APU (gfx1151)
- ROCm 7.2+ drivers installed on the host
- Docker with BuildKit enabled

## Quick Start

```bash
cd docker/rocm-gfx1151
docker compose up --build
```

The service will be available at `http://localhost:8880`.

## MIOpen Cache Warmup (Recommended)

MIOpen compiles optimized GPU kernels for each unique tensor shape. On gfx1151, this means a 5-60 second delay on the first request for each phoneme length. To eliminate this:

1. Start the container with `MIOPEN_FIND_MODE=3` to force kernel search:
   ```bash
   docker compose run -e MIOPEN_FIND_MODE=3 -e MIOPEN_FIND_ENFORCE=3 kokoro-tts
   ```

2. Run the warmup script inside the container:
   ```bash
   docker exec -it <container> bash -lc 'python /app/docker/rocm/warmup_miopen.py'
   ```

3. This takes ~2 hours on Strix Halo but only needs to run once per ROCm/PyTorch upgrade.

4. Restart the container normally (without the `MIOPEN_FIND_MODE=3` override). The default `MIOPEN_FIND_MODE=2` will reuse the cached kernels.

## Environment Variables

These are set in the Dockerfile by default:

| Variable | Value | Purpose |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | `11.5.1` | Required because ROCm does not officially list gfx1151 |
| `HIP_VISIBLE_DEVICES` | `0` | Pins to the first (typically only) GPU |
| `MIOPEN_FIND_MODE` | `2` | Reuses on-disk MIOpen cache |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | Enables experimental attention ops |

Override any of these in `docker-compose.yml` if needed.

## Building Manually

```bash
docker build -f docker/rocm-gfx1151/Dockerfile -t kokoro-fastapi-rocm-gfx1151:latest .
```

## Troubleshooting

### "No HIP devices found"
- Ensure `/dev/kfd` and `/dev/dri` are passed through (done by default in compose)
- Verify `rocminfo` on the host shows `gfx1151`
- Check that `HSA_OVERRIDE_GFX_VERSION=11.5.1` is set

### Slow first requests
- This is normal without a warmed MIOpen cache. Run the warmup script above.

### Container crashes on startup
- Ensure `ipc: host` is set in docker-compose.yml (PyTorch shared memory requirement)
