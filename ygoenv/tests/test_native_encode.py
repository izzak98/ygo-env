import numpy as np
import pytest

try:
    import ygoenv.ygopro  # noqa: F401
    HAS_SO = True
except ImportError:
    HAS_SO = False

from ygoenv.modes import _native_has_ml_obs
from ygoenv.paths import embeddings_path


pytestmark = pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")


def _stub_embeddings():
    from ygoenv.decoding import get_code_list

    z = [0.0] * 8
    return {c: {"embedding": z, "name_embedding": z} for c in get_code_list()}


def _load_embeddings():
    from ygoenv.env_wrapping.interface import load_embeddings

    if embeddings_path().is_file():
        return load_embeddings()
    return _stub_embeddings()


def test_ygoenv_raw_hides_ml_keys():
    from ygoenv import GameMode, YGOEnv

    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck="tear",
        num_envs=1,
        seed_mode="full_det",
        base_seed=42,
        obs_format="raw",
    )
    try:
        obs = env.obs
        assert "cards_" in obs
        assert "actions_" in obs
        assert all(not k.startswith("ml_") for k in obs)
        assert all(isinstance(v, np.ndarray) for v in obs.values())
    finally:
        env.close()


@pytest.mark.skipif(not _native_has_ml_obs(), reason="native ML obs keys missing")
def test_ygoenv_vectorized_is_numpy_ml_keys():
    from ygoenv import GameMode, YGOEnv

    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck="tear",
        num_envs=1,
        seed_mode="full_det",
        base_seed=42,
        obs_format="vectorized",
    )
    try:
        obs = env.obs
        assert "cards_" not in obs
        for key in (
            "ml_card_emb_idx_",
            "ml_hist_emb_idx_",
            "ml_card_static_",
            "ml_card_dynamic_",
            "ml_history_info_",
            "ml_prompt_",
            "ml_n_me_",
        ):
            assert key in obs
            assert isinstance(obs[key], np.ndarray)
        assert obs["ml_card_static_"].shape[-1] == 85
        assert obs["ml_card_dynamic_"].shape[-1] == 26
        assert obs["ml_prompt_"].shape[-1] == 13
        n_me = int(np.asarray(obs["ml_n_me_"]).reshape(-1)[0])
        assert 0 <= n_me <= obs["ml_card_emb_idx_"].shape[-1]
    finally:
        env.close()


@pytest.mark.skipif(not _native_has_ml_obs(), reason="native ML obs keys missing")
def test_vectorized_rejected_for_pvp():
    from ygoenv import GameMode, YGOEnv

    with pytest.raises(ValueError, match="play_vs_opponent"):
        YGOEnv(
            mode=GameMode.PLAY_VS_OPPONENT,
            deck="tear",
            num_envs=1,
            obs_format="vectorized",
        )


@pytest.mark.skipif(not _native_has_ml_obs(), reason="native ML obs keys missing")
def test_native_compact_matches_encode_all_batch_fast():
    from ygoenv import GameMode, YGOEnv
    from ygoenv.env_wrapping.interface import encode_all_batch_fast
    from ygoenv.constants import STATIC_INTRINSIC_DIM

    embeddings = _load_embeddings()
    from ygoenv.decoding import get_code_list
    from ygoenv.env_wrapping.interface import _embedding_cache
    from ygoenv.ygopro import set_emb_index_map

    _embedding_cache.init_fast(embeddings, get_code_list())
    set_emb_index_map(_embedding_cache._code_to_emb_idx.tolist())

    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck="tear",
        num_envs=1,
        seed_mode="full_det",
        base_seed=42,
        obs_format="vectorized",
    )
    rng = np.random.default_rng(0)
    try:
        for _ in range(5):
            raw = env._obs
            encoded, prompt = encode_all_batch_fast(
                raw, embeddings, use_preallocated_gpu=False
            )
            ml = env.obs
            n_me = int(np.asarray(ml["ml_n_me_"]).reshape(-1)[0])
            py_emb = encoded.card_emb_idx.detach().cpu().numpy()
            nat_emb = ml["ml_card_emb_idx_"].astype(np.int64)
            assert nat_emb.shape == py_emb.shape
            np.testing.assert_array_equal(nat_emb, py_emb)

            py_board = encoded.card_static_board.detach().cpu().numpy()
            nat_board = ml["ml_card_static_"][..., STATIC_INTRINSIC_DIM:]
            np.testing.assert_allclose(nat_board, py_board, atol=1e-5)

            py_dyn = encoded.card_dynamic.detach().cpu().numpy()
            np.testing.assert_allclose(ml["ml_card_dynamic_"], py_dyn, atol=1e-5)

            py_hist = encoded.history_info.detach().cpu().numpy()
            np.testing.assert_allclose(ml["ml_history_info_"], py_hist, atol=1e-5)

            py_h_emb = encoded.hist_emb_idx.detach().cpu().numpy()
            np.testing.assert_array_equal(
                ml["ml_hist_emb_idx_"].astype(np.int64), py_h_emb
            )

            py_prompt = prompt.detach().cpu().numpy()
            np.testing.assert_allclose(ml["ml_prompt_"], py_prompt, atol=1e-5)

            n_opt = int(np.any(raw["actions_"][0] != 0, axis=1).sum())
            act = 0 if n_opt <= 0 else int(rng.integers(0, n_opt))
            _, _, terminated, _, _ = env.step(np.array([act], dtype=np.int32))
            if terminated[0]:
                env.reset()
    finally:
        env.close()


@pytest.mark.skipif(not _native_has_ml_obs(), reason="native ML obs keys missing")
@pytest.mark.skipif(
    not embeddings_path().is_file(),
    reason=f"embeddings.json not found at {embeddings_path()}",
)
def test_env_wrapper_vectorized_default_smoke():
    from ygoenv import EnvWrapper

    wrapper = EnvWrapper(
        deck="tear",
        num_envs=1,
        base_seed=42,
        mode="full_det",
        auto_reset=False,
    )
    try:
        assert wrapper.obs_format == "vectorized"
        assert wrapper.obs.card_emb_idx.shape[0] == 1
        wrapper.step(np.array([0], dtype=np.int32))
    finally:
        wrapper.close()
