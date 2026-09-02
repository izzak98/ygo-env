"""Lazy vocabulary-index → intrinsic static row table (dims 0..STATIC_INTRINSIC_DIM-1).

Rows are populated from live ``encode_all_batch_fast`` output the first time an
embedding index appears, so values match byte-accurate encoding. Unknown indices
at ``gather`` time (e.g. stale replay after restart) yield zeros."""

from __future__ import annotations

import threading
from typing import Any, Dict

import numpy as np
import torch

from ygoenv.constants import STATIC_INTRINSIC_DIM


class LazyIntrinsicStaticTable:
    def __init__(self) -> None:
        self._rows: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def update_from_numpy(
        self,
        card_emb_idx_np: np.ndarray,
        card_static_np: np.ndarray,
    ) -> None:
        """Register intrinsic rows for newly seen ``card_emb_idx`` values."""
        valid = card_emb_idx_np >= 0
        if not valid.any():
            return
        flat_e = card_emb_idx_np[valid].ravel()
        intr = card_static_np[valid][:, :STATIC_INTRINSIC_DIM]
        with self._lock:
            for i in range(flat_e.shape[0]):
                e = int(flat_e[i])
                if e not in self._rows:
                    self._rows[e] = intr[i].astype(np.float32, copy=True)

    def gather(
        self,
        card_emb_idx: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """(B, N, STATIC_INTRINSIC_DIM); padding / unknown idx → zeros."""
        b, n = card_emb_idx.shape
        flat = card_emb_idx.reshape(-1)
        out = torch.zeros(flat.shape[0], STATIC_INTRINSIC_DIM, device=device, dtype=dtype)
        valid = flat >= 0
        if not valid.any():
            return out.view(b, n, STATIC_INTRINSIC_DIM)
        vals = flat[valid].detach().cpu().numpy()
        with self._lock:
            stacked = [
                self._rows[int(e)] if int(e) in self._rows
                else np.zeros(STATIC_INTRINSIC_DIM, dtype=np.float32)
                for e in vals
            ]
        mat = torch.from_numpy(np.stack(stacked, axis=0)).to(device=device, dtype=dtype)
        out[valid] = mat
        return out.view(b, n, STATIC_INTRINSIC_DIM)

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {str(k): v.copy() for k, v in self._rows.items()}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        with self._lock:
            self._rows.clear()
            for k, v in d.items():
                self._rows[int(k)] = np.asarray(v, dtype=np.float32)


_table = LazyIntrinsicStaticTable()


def get_lazy_intrinsic_table() -> LazyIntrinsicStaticTable:
    return _table
