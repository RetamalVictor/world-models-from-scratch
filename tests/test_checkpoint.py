import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training import train_state

from world_models.checkpoint import Checkpointer


def _state(value: float):
    return train_state.TrainState.create(
        apply_fn=lambda p, x: x,
        params={"w": jnp.full(3, value)},
        tx=optax.adam(1e-3),
    )


def _tree(value: float, round_: int):
    return {"model": _state(value), "counters": {"round": round_}}


def test_roundtrip_restores_params_counters_and_opt_state(tmp_path):
    ckpt = Checkpointer(tmp_path)
    saved = _tree(1.5, 7)
    ckpt.save(7, saved)

    step, restored = ckpt.restore(_tree(0.0, 0))
    assert step == 7
    assert restored["counters"]["round"] == 7
    np.testing.assert_array_equal(restored["model"].params["w"],
                                  saved["model"].params["w"])
    # optimizer state comes back too — a resumed run is the same run
    np.testing.assert_array_equal(
        restored["model"].opt_state[0].mu["w"],
        saved["model"].opt_state[0].mu["w"])


def test_keeps_last_k_and_latest_wins(tmp_path):
    ckpt = Checkpointer(tmp_path, keep=2)
    for s in (1, 2, 3, 4):
        ckpt.save(s, _tree(float(s), s))
    assert ckpt.steps() == [3, 4]
    step, restored = ckpt.restore(_tree(0.0, 0))
    assert step == 4
    assert float(restored["model"].params["w"][0]) == 4.0


def test_best_survives_pruning(tmp_path):
    ckpt = Checkpointer(tmp_path, keep=1)
    for s, metric in ((1, 0.1), (2, 0.9), (3, 0.4)):
        ckpt.save(s, _tree(float(s), s), metric=metric)
    assert ckpt.steps() == [3]
    assert ckpt.best_meta() == {"step": 2, "metric": 0.9}
    step, best = ckpt.restore_best(_tree(0.0, 0))
    assert step == 2
    assert float(best["model"].params["w"][0]) == 2.0


def test_restore_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Checkpointer(tmp_path).restore(_tree(0.0, 0))
