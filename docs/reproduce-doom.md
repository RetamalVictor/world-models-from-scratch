# Reproducing the take_cover results

Commands to reproduce the take_cover survival numbers: random 70.4,
plain loop 105.7, imagination temperature 151.3, replay bundle 152.9,
composed 186.8 (all n=100, 95k env frames). Source record: journals
[0011](journal/0011-runpod-full-loop.md),
[0012](journal/0012-imagination-temperature.md),
[0013](journal/0013-replay-bundle.md), and the ladder in
[design/sample-efficiency.md](design/sample-efficiency.md).

## Requirements

- Linux with an NVIDIA GPU, 12 GB is plenty (peak VRAM at 64x64 is
  about 1.1 GiB, per journal 0010). On Windows, run this inside WSL2:
  JAX has no native Windows CUDA path.
- `uv sync --extra doom --extra cuda`
- wandb is optional (`uv sync --extra wandb`). Every command below
  works without it: if the extra is not installed, the tracker
  catches the import failure and falls back to the local jsonl only.
  To skip the mirror explicitly, add `--no-wandb`.
- Rough wall-clock and cost on a rented RTX 4090: the pretrain (stage
  a below) takes about 1 hour; the composed loop (stage b) about 3.5
  hours. Journal 0011's plain pretrain+loop pair (no temperature, no
  bundle) cost about $4 total on the rented pod; the composed run
  does 4x the actor-critic updates per round, so budget somewhat
  more, still single-digit dollars at typical 4090 rental prices.
  Evaluation and diagnostics only run the frozen model, no training:
  a few minutes each.

## The headline result

All commands below write into `runs/doom/pretrain` (the run root is
`runs/doom` by default; there is no CLI flag for it).

### a. World-model-only pretrain (about 1 hour)

    uv run train-doom --wm-only --seed-episodes 1000 \
        --episodes-per-round 0 --rounds 200 --run-name pretrain

1000 random episodes (73,061 frames), 200 rounds of 100 world-model
updates each, no actor-critic. Expect, by round 200: recon 1.49, kl
3.45, continue_bce 0.007, with kl noisy in a 3.5-5 band through
training and continue_bce spiking on death-carrying batches (journal
0011). The run's own built-in eval reports a random baseline around
77.7 steps at the end, but that is an n=3 estimate, treat it as a
sanity check only, not the n=100 random number used below.

### b. Composed loop, resumed in the same run dir (about 3.5 hours)

    uv run train-doom --resume --run-name pretrain --rounds 500 \
        --imagination-temperature 1.30 --ac-updates 400 \
        --ac-reset-every 100 --critic-ema 0.98 --final-ac-steps 5000

300 more rounds (201-500). Per round: one on-policy episode collected
by the sampled actor, 100 world-model updates, 400 actor-critic
updates in imagination at tau 1.30, lambda-returns bootstrapped from
a 0.98 EMA copy of the critic. Actor and critic reinitialize at
rounds 300 and 400 (never on the final round). A 5000-step fresh
refit runs after round 500 into the `actor_refit` arm, which the
headline number below does not use. Total env frames at the end:
about 95k (73k random plus about 22k on-policy).

### c. Evaluation at n=100

    uv run python scripts/doom_eval.py --run runs/doom/pretrain --episodes 100

Expect, n=100, 95k env frames:

| arm | steps | tics | vs random |
|---|---|---|---|
| random | 70.4 +/- 32.4 | 282 | 1.00x |
| actor (composed, co-trained) | 186.8 +/- 128.2 | 747 | 2.65x |

The random arm is seeded independently of the trained model (default
seed 0), so it reproduces exactly across runs; the actor arm is where
GPU nondeterminism shows up (see below).

### d. Diagnostics (optional)

    uv run python scripts/doom_diag.py --run runs/doom/pretrain

Checks whether the continue head's death signal lives somewhere
imagination can reach. Point it at the stage-a checkpoint, before any
actor gradient: from the pod pretrain in journal 0011, expect the
one-step prior test at the terminal frame to have posterior median
0.001 (89% below 0.5), anticipation (posterior one step before death)
median 0.012 (81% below 0.5), 83% of dream rollouts dipping below
0.5, and 21% of calm frames (20+ steps from any death) scoring below
0.5, the honest cost of a head trained eager rather than blind.

## Reproducibility caveat

GPU matmul reduction order and kernel autotuning are not fixed by
default, so a rerun matches the numbers above statistically (same
seed, same distribution), not bit-for-bit. Journal 0011 pins matmul
precision and autotuning so its own reruns are comparable; the two
env vars that do this in JAX/XLA are:

    JAX_DEFAULT_MATMUL_PRECISION=highest
    XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"

Set both before `uv run train-doom ...`. Even with these set, the
ViZDoom engine's own state cannot be checkpointed, so a `--resume`
plays different on-policy episodes than an uninterrupted run would
have (the training math on a given buffer is still bit-reproducible,
per the module docstring). Expect a single rerun to land within the
reported std, not on the exact mean.

## Ablation table: reproducing one row at a time

`--resume` advances the run dir's counters and checkpoints in place,
so getting more than one row out of the same pretrain means copying
the run dir first:

    cp -r runs/doom/pretrain runs/doom/<variant>

then resuming `<variant>` with the row's flags, rounds 500 in every
case:

| row | survival, n=100 | flags added to the plain resume |
|---|---|---|
| plain loop | 105.7 +/- 65.5 | (none) |
| + imagination temperature | 151.3 +/- 99.2 | `--imagination-temperature 1.30` |
| + replay bundle | 152.9 +/- 77.1 | `--ac-updates 400 --ac-reset-every 100 --critic-ema 0.98` |
| + both (composed) | 186.8 +/- 128.2 | all four flags together |

`--final-ac-steps 5000` is optional in every row: it produces the
`actor_refit` arm for `doom_eval.py` but does not change the
co-trained `actor` numbers above, since the refit trains only in
imagination after round 500 and collects no new env frames. The
plain-loop row's exact command:

    uv run train-doom --resume --run-name loop-plain --rounds 500

## What is in a run dir

- `config.json`: the full training config, plus git commit, JAX
  backend, and device list.
- `metrics.jsonl`: one line per round (recon, kl, continue_bce,
  env_frames, actor_loss, critic_loss, imag_reward, real_steps at
  eval rounds).
- `checkpoints/step_00000200.msgpack` and so on: rotating, keeps the
  last 3 (not a CLI flag); plus `best.msgpack` and `best.json`, kept
  outside the rotation, the checkpoint with the highest real-eval
  survival seen so far.
- `buffer.npz`: the replay buffer, saved alongside every checkpoint.
- `checkpoint.msgpack`, `actor.msgpack`, `actor_refit.msgpack`: final
  world-model and actor params, written once at the end of training.
- `dream.gif`, `dream_filmstrip.png`: a sampled-prior rollout from a
  real warmup, next to the true continuation.
- `loss_curves.png`: the curves in `metrics.jsonl`, plotted.
- `eval.json`: the training run's own built-in eval (n=3 by default).
- `eval_big.json`, `diag.json`: written by `doom_eval.py` and
  `doom_diag.py` into the run dir unless `--out` points elsewhere.
