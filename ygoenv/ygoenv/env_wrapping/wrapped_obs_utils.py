"""Expand compact observations (embedding indices) to full encoder tensors."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ygoenv.constants import DEVICE, HISTORY_INFO_DIM, STATIC_CARD_DIM
from ygoenv.env_wrapping.interface import EncodedObsCompact, _embedding_cache, load_embeddings
from ygoenv.env_wrapping.lazy_intrinsic_static import get_lazy_intrinsic_table


def ensure_embedding_cache() -> None:
    """Lazy-init vocabulary tables for gather (e.g. replay before any env encode)."""
    if _embedding_cache._embeddings_dict is None:
        _embedding_cache.initialize(load_embeddings())


def reconstruct_full_card_static(
    card_emb_idx: torch.Tensor,
    card_static_board: torch.Tensor,
) -> torch.Tensor:
    """Concat lazy intrinsic rows (by emb idx) with per-slot board columns → (B,N,STATIC_CARD_DIM)."""
    device = card_static_board.device
    intrinsic = get_lazy_intrinsic_table().gather(
        card_emb_idx, device=device, dtype=torch.float32
    )
    board = card_static_board
    if board.dtype in (torch.float16, torch.bfloat16):
        board = board.float()
    return torch.cat([intrinsic, board], dim=-1)


def gather_pretrained_embeddings(
    card_emb_idx: torch.Tensor,
    hist_emb_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather (B,N,1024) name/desc and (B,H,1024) hist from vocabulary tables."""
    ensure_embedding_cache()
    name_t, desc_t = _embedding_cache.get_gpu_tables()
    if name_t is None:
        name_t = torch.from_numpy(_embedding_cache._name_embedding_array).to(
            device=DEVICE, dtype=torch.float32
        )
        desc_t = torch.from_numpy(_embedding_cache._embedding_array).to(
            device=DEVICE, dtype=torch.float32
        )
    valid_c = card_emb_idx >= 0
    valid_h = hist_emb_idx >= 0
    valid_c = valid_c.to(DEVICE)
    valid_h = valid_h.to(DEVICE)
    s_c = card_emb_idx.clamp(min=0).to(torch.int64)
    s_h = hist_emb_idx.clamp(min=0).to(torch.int64)
    cn = name_t[s_c]
    cd = desc_t[s_c]
    cn = cn * valid_c.unsqueeze(-1).to(cn.dtype)
    cd = cd * valid_c.unsqueeze(-1).to(cd.dtype)
    hn = name_t[s_h]
    hd = desc_t[s_h]
    hn = hn * valid_h.unsqueeze(-1).to(hn.dtype)
    hd = hd * valid_h.unsqueeze(-1).to(hd.dtype)
    return cn, cd, hn, hd


# Keys required by ``Encoder.forward`` / ``MultiStreamEncoder.forward`` (history included).
_ENCODER_CORE_KEYS = (
    "card_name_emb",
    "card_desc_emb",
    "card_static",
    "card_dynamic",
    "history_name_emb",
    "history_desc_emb",
    "history_info",
)


def _validate_encoder_tensor_dict(d: Dict[str, torch.Tensor]) -> None:
    """Assert batch / slot shapes align so history cannot be dropped silently."""
    missing = [k for k in _ENCODER_CORE_KEYS if k not in d]
    if missing:
        raise KeyError(f"encoder input dict missing keys: {missing}")
    cn = d["card_name_emb"]
    hn = d["history_name_emb"]
    hd = d["history_desc_emb"]
    hi = d["history_info"]
    B, Nc = cn.shape[0], cn.shape[1]
    if d["card_desc_emb"].shape[:2] != (B, Nc):
        raise ValueError(
            f"card_desc_emb shape {tuple(d['card_desc_emb'].shape)} != card_name_emb (B,N)=({B},{Nc})"
        )
    if d["card_static"].shape[:2] != (B, Nc) or d["card_dynamic"].shape[:2] != (B, Nc):
        raise ValueError(
            f"card_static / card_dynamic must be (B={B}, N_cards={Nc}, *), got "
            f"{tuple(d['card_static'].shape)}, {tuple(d['card_dynamic'].shape)}"
        )
    if hn.shape[:2] != hd.shape[:2]:
        raise ValueError(
            f"history_name_emb {tuple(hn.shape)} vs history_desc_emb {tuple(hd.shape)}"
        )
    if hn.shape[0] != B:
        raise ValueError(
            f"history batch {hn.shape[0]} != card batch {B}"
        )
    Nh = hn.shape[1]
    if hi.shape[:2] != (B, Nh) or hi.shape[-1] != HISTORY_INFO_DIM:
        raise ValueError(
            f"history_info expected (*, *, {HISTORY_INFO_DIM}) with batch (B,N_hist)=({B},{Nh}), "
            f"got {tuple(hi.shape)}"
        )


def encoder_tensor_dict_from_compact(
    compact: EncodedObsCompact,
    embeddings: dict,
    *,
    prompt_one_hot: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build keyword args for MultiStreamEncoder.forward."""
    _ = embeddings
    cn, cd, hn, hd = gather_pretrained_embeddings(
        compact.card_emb_idx, compact.hist_emb_idx
    )
    cs = reconstruct_full_card_static(compact.card_emb_idx, compact.card_static_board)
    d = {
        "card_name_emb": cn,
        "card_desc_emb": cd,
        "card_static": cs,
        "card_dynamic": compact.card_dynamic,
        "history_name_emb": hn,
        "history_desc_emb": hd,
        "history_info": compact.history_info,
        "prompt_one_hot": prompt_one_hot,
    }
    _validate_encoder_tensor_dict(d)
    return d


def encoder_tensor_dict_from_wrapped_obs(
    obs: Any,
    embeddings: dict | None = None,
) -> Dict[str, torch.Tensor]:
    """Expand compact WrappedObs to full encoder tensors (cards + history streams)."""
    ensure_embedding_cache()
    if getattr(obs, "card_emb_idx", None) is not None:
        if getattr(obs, "hist_emb_idx", None) is None or getattr(obs, "history_info", None) is None:
            raise TypeError(
                "WrappedObs must include hist_emb_idx and history_info for encoder inputs"
            )
        cn, cd, hn, hd = gather_pretrained_embeddings(
            obs.card_emb_idx, obs.hist_emb_idx
        )
        if getattr(obs, "card_static_board", None) is not None:
            cs = reconstruct_full_card_static(obs.card_emb_idx, obs.card_static_board)
        elif getattr(obs, "card_static", None) is not None:
            cs = obs.card_static
            if cs.shape[-1] != STATIC_CARD_DIM:
                raise ValueError("card_static must have last dim STATIC_CARD_DIM")
        else:
            raise TypeError("WrappedObs needs card_static_board or card_static")
        return {
            "card_name_emb": cn,
            "card_desc_emb": cd,
            "card_static": cs,
            "card_dynamic": obs.card_dynamic,
            "history_name_emb": hn,
            "history_desc_emb": hd,
            "history_info": obs.history_info,
            # "prompt_one_hot": obs.prompt_one_hot,
        }
    raise TypeError("obs must be compact WrappedObs with card_emb_idx")


def _promote_stored_floats(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.float()
    return t


def expand_obs_for_encoder(
    obs: Any,
    embeddings: dict | None = None,
) -> Dict[str, torch.Tensor]:
    """Build keyword args for ``encoder.forward`` including gathered history embeddings and ``history_info``."""
    d = encoder_tensor_dict_from_wrapped_obs(obs, embeddings)
    for k in ("card_static", "card_dynamic", "history_info", ):  # "prompt_one_hot"
        if k in d:
            d[k] = _promote_stored_floats(d[k])
    _validate_encoder_tensor_dict(d)
    return d


def encoder_forward_kwargs(
    obs: Any,
    *,
    device: torch.device,
    embeddings: dict | None = None,
) -> Dict[str, torch.Tensor]:
    """Same as :func:`expand_obs_for_encoder`, then move all tensors to ``device`` (for ``encoder.forward``)."""
    d = expand_obs_for_encoder(obs, embeddings)
    non_blocking = device.type == "cuda"
    return {k: v.to(device, non_blocking=non_blocking) for k, v in d.items()}
