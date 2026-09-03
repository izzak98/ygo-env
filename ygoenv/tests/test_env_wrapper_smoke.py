import numpy as np
import pytest

try:
    import ygoenv.ygopro  # noqa: F401
    HAS_SO = True
except ImportError:
    HAS_SO = False

from ygoenv.constants import STATIC_CARD_DIM, STATIC_INTRINSIC_DIM, STATIC_BOARD_DIM
from ygoenv.env_wrapping.wrapped_obs_utils import reconstruct_full_card_static
from ygoenv.paths import embeddings_path


def test_reconstruct_full_card_static_shape():
    import torch
    b, n = 2, 4
    card_emb_idx = torch.tensor([[0, 1, -1, 2], [3, -1, 4, 5]], dtype=torch.int64)
    card_static_board = torch.zeros(b, n, STATIC_BOARD_DIM)
    out = reconstruct_full_card_static(card_emb_idx, card_static_board)
    assert out.shape == (b, n, STATIC_CARD_DIM)
    assert out.dtype == torch.float32
    assert out[0, 2].abs().max().item() == 0.0


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.skipif(
    not embeddings_path().is_file(),
    reason=f"embeddings.json not found at {embeddings_path()}",
)
def test_env_wrapper_smoke_tear():
    from ygoenv import EnvWrapper, ENV_MODES

    wrapper = EnvWrapper(
        deck="tear",
        num_envs=1,
        base_seed=42,
        mode=ENV_MODES[0],
        auto_reset=False,
    )
    obs = wrapper.obs
    assert obs.card_emb_idx.shape[0] == 1
    assert obs.card_static_board.shape[-1] == STATIC_BOARD_DIM
    n_opt = int(np.any(wrapper._obs["actions_"][0] != 0, axis=1).sum())
    assert n_opt >= 1
    wrapper.step(np.array([0], dtype=np.int32))
    wrapper.close()


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.skipif(
    not embeddings_path().is_file(),
    reason=f"embeddings.json not found at {embeddings_path()}",
)
def test_env_wrapper_smoke_maliss():
    from ygoenv import EnvWrapper

    wrapper = EnvWrapper(deck="maliss", num_envs=1, auto_reset=False)
    assert wrapper.obs.card_emb_idx.shape[0] == 1
    wrapper.close()
