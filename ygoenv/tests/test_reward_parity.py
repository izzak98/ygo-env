"""Verify that native shaped_first_credit reward values are finite across random steps."""

import numpy as np
import pytest

try:
    import ygoenv.ygopro  # noqa: F401
    HAS_SO = True
except ImportError:
    HAS_SO = False


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.parametrize("deck", ["tear", "ryzeal"])
def test_native_reward_is_finite(deck):
    """shaped_first_credit rewards should all be finite floats."""
    from ygoenv import GameMode, YGOEnv

    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck=deck,
        num_envs=1,
        seed_mode="full_det",
        base_seed=42,
        reward_mode="shaped_first_credit",
        episode_done_mode="turn",
    )
    obs = env.reset()
    rng = np.random.default_rng(0)
    try:
        for _ in range(8):
            n_opt = int(np.any(obs["actions_"][0] != 0, axis=1).sum())
            act = 0 if n_opt <= 0 else int(rng.integers(0, n_opt))
            obs, rews, terminated, truncated, _ = env.step(np.array([act], dtype=np.int32))
            assert np.isfinite(rews[0]), f"reward {rews[0]} is not finite"
            done = bool(terminated[0]) or bool(truncated[0])
            if done:
                obs = env.reset()
    finally:
        env.close()


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.parametrize("deck", ["tear", "ryzeal"])
def test_terminal_board_value_zero_mid_episode(deck):
    """terminal_board_value: all rewards before done are 0."""
    from ygoenv import GameMode, YGOEnv

    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck=deck,
        num_envs=1,
        seed_mode="full_det",
        base_seed=42,
        reward_mode="terminal_board_value",
        episode_done_mode="turn",
    )
    obs = env.reset()
    rng = np.random.default_rng(0)
    try:
        for _ in range(6):
            n_opt = int(np.any(obs["actions_"][0] != 0, axis=1).sum())
            act = 0 if n_opt <= 0 else int(rng.integers(0, n_opt))
            obs, rews, terminated, truncated, _ = env.step(np.array([act], dtype=np.int32))
            done = bool(terminated[0]) or bool(truncated[0])
            if not done:
                assert rews[0] == 0.0, f"expected 0 mid-episode, got {rews[0]}"
            else:
                assert np.isfinite(rews[0])
                break
    finally:
        env.close()
