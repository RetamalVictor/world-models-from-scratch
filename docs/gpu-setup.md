# GPU training (RTX 5070 Ti via WSL2)

JAX has no native Windows CUDA support, so GPU training runs inside WSL2.
The Windows side stays the editing/dev environment with CPU JAX; the
Ubuntu side owns the GPU runs. Every run records which backend it used in
its `config.json` (`jax_backend`), so a CPU run can never masquerade as a
GPU one in the results.

## One-time setup (done)

- WSL2 with Ubuntu-24.04 — installed.
- uv inside Ubuntu — installed.
- Linux-side clone at `~/world_models`, synced with `--extra cuda` so the
  CUDA wheels are already downloaded.

## Per-session flow

1. Activate the GPU (manual step on this machine). Check it took:

   ```
   wsl -d Ubuntu-24.04 -- nvidia-smi
   ```

   If this fails with an NVML error, the GPU is not active yet.

2. Pull the latest commits into the Linux clone. The Windows repo is
   reachable from WSL, no remote needed:

   ```
   wsl -d Ubuntu-24.04
   cd ~/world_models
   git pull origin <branch>        # origin points at /mnt/c/.../world_models
   uv sync --extra cuda
   ```

3. Verify JAX sees the card before launching anything long:

   ```
   uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
   ```

   Expect `gpu [CudaDevice(id=0)]`. If it says `cpu`, stop and fix that
   first; a silent CPU fallback would poison timing comparisons.

4. Regenerate data and train:

   ```
   uv run make-data          # deterministic, no need to copy files over
   uv run train-vae --run-name <name>
   ```

Run artifacts stay in the Linux clone under `runs/`. Copy anything worth
keeping into the journal, or pull it back to the Windows side via
`\\wsl$\Ubuntu-24.04\home\victo\world_models\runs`.

## Notes

- Only commits travel between the two clones (via git), never loose
  files. If a result matters, its code state is committed first.
- The dataset is regenerated per clone from seed 0, byte-identical by
  construction; `data/ball.stats.json` should match across clones and is
  the quick check that it did.
- The 5070 Ti is a Blackwell card: if `jax[cuda12]` ever refuses it,
  the fix is moving the `cuda` extra to the newer CUDA wheel line, not
  pinning old drivers.
