import pytest

try:
    import ygoenv.ygopro  # noqa: F401
    HAS_SO = True
except ImportError:
    HAS_SO = False


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
def test_ygo_env_board_setup_reset_step():
    from ygoenv import GameMode, YGOEnv

    env = YGOEnv(mode=GameMode.BOARD_SETUP, deck="tear", num_envs=1, seed_mode="full_det", base_seed=42)
    obs = env.reset()
    assert "cards_" in obs
    assert len(env.decoded_cards) == 1
    env.close()
