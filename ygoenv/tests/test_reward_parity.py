import numpy as np
import pytest

try:
    import ygoenv.ygopro  # noqa: F401
    HAS_SO = True
except ImportError:
    HAS_SO = False


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.parametrize("deck", ["tear", "ryzeal"])
def test_native_deck_reward_matches_python(deck):
    from ygoenv import GameMode, YGOEnv
    from ygoenv.modes import _legacy_native_build
    from ygoenv.rewards import get_reward

    if _legacy_native_build():
        pytest.skip("native deck rewards require a current ygopro_ygoenv.so build")

    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck=deck,
        num_envs=1,
        seed_mode="full_det",
        base_seed=42,
    )
    obs = env.reset()
    rng = np.random.default_rng(0)
    try:
        for _ in range(8):
            n_opt = int(np.any(obs["actions_"][0] != 0, axis=1).sum())
            act = 0 if n_opt <= 0 else int(rng.integers(0, n_opt))
            obs, rews, dones, done_idx, _ = env.step(np.array([act], dtype=np.int32))
            cards = list(env.decoded_cards[0])
            py_r = float(get_reward(deck, cards))
            assert rews[0] == pytest.approx(py_r, abs=1e-5)
            if len(done_idx) > 0:
                obs = env.reset(env_indices=done_idx.tolist())
    finally:
        env.close()
