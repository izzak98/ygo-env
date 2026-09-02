import os

import numpy as np
import pytest

try:
    import ygoenv.ygopro  # noqa: F401
    HAS_SO = True
except ImportError:
    HAS_SO = False


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
def test_ygo_env_board_setup_reset_step():
    from ygoenv import GameMode, YGOEnv

    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck="tear",
        num_envs=1,
        seed_mode="full_det",
        base_seed=42,
    )
    obs = env.reset()
    assert "cards_" in obs
    assert len(env.decoded_cards) == 1
    n_opt = int(np.any(obs["actions_"][0] != 0, axis=1).sum())
    assert n_opt >= 1
    obs, rews, dones, done_idx, raw = env.step(np.array([0], dtype=np.int32))
    assert len(rews) == 1
    env.close()


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
def test_init_ygopro_from_non_repo_cwd(tmp_path):
    from ygoenv.init import init_ygopro

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        name = init_ygopro("tear", opponent_deck=None)
        assert name == "tear"
    finally:
        os.chdir(old)
