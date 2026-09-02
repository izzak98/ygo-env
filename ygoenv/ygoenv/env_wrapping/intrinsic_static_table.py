"""Build per-vocabulary intrinsic static features (first STATIC_INTRINSIC_DIM dims) from SQLite."""

from __future__ import annotations

import os
import sqlite3
from typing import TYPE_CHECKING, Optional

import numpy as np

from ygoenv.constants import STATIC_INTRINSIC_DIM
from ygoenv.decoding import CardRecord, ATTRIBUTE_STRS, RACE_STRS
from ygoenv.env_wrapping.interface import _encode_cards_vectorized

if TYPE_CHECKING:
    pass

# YGOPRO common.h TYPE_* -> decoding.TYPE_STRS order (24 entries)
_YGO_TYPE_MASKS: list[tuple[int, str]] = [
    (0x1, "Monster"),
    (0x2, "Spell"),
    (0x4, "Trap"),
    (0x10, "Normal"),
    (0x20, "Effect"),
    (0x40, "Fusion"),
    (0x80, "Ritual"),
    (0x100, "Trap Monster"),
    (0x200, "Spirit"),
    (0x400, "Union"),
    (0x800, "Dual"),
    (0x1000, "Tuner"),
    (0x2000, "Synchro"),
    (0x4000, "Token"),
    (0x10000, "Quick-play"),
    (0x20000, "Continuous"),
    (0x40000, "Equip"),
    (0x80000, "Field"),
    (0x100000, "Counter"),
    (0x200000, "Flip"),
    (0x400000, "Toon"),
    (0x800000, "XYZ"),
    (0x1000000, "Pendulum"),
    (0x2000000, "Special"),
    (0x4000000, "Link"),
]


def _type_int_to_types(type_int: int) -> list[str]:
    out: list[str] = []
    for mask, name in _YGO_TYPE_MASKS:
        if type_int & mask:
            out.append(name)
    return out


def _attr_int_to_str(attr_int: int) -> Optional[str]:
    if attr_int == 0:
        return None
    for i, name in enumerate(ATTRIBUTE_STRS):
        bit = 1 << i
        if attr_int & bit:
            return name
    return None


def _race_int_to_str(race_int: int) -> Optional[str]:
    if race_int == 0:
        return None
    for i, name in enumerate(RACE_STRS):
        bit = 1 << i
        if race_int & bit:
            return name
    return None


def _datas_row_to_cardrecord(
    card_id: str,
    type_int: int,
    atk: int,
    def_: int,
    level: int,
    race_int: int,
    attr_int: int,
) -> CardRecord:
    types = _type_int_to_types(type_int)
    is_monster = "Monster" in types
    is_pendulum = "Pendulum" in types
    level_packed = int(level)
    level_star = level_packed & 0xFF
    lscale_db = (level_packed >> 24) & 0xFF

    return CardRecord(
        card_id=card_id,
        card_index=None,
        location="Hand",
        seq=None,
        owner="me",
        position=None,
        overlay=False,
        attribute=_attr_int_to_str(attr_int) if is_monster else None,
        race=_race_int_to_str(race_int) if is_monster else None,
        level=level_star if is_monster and level_star else None,
        counter=None,
        negated=False,
        atk_raw=int(atk) if is_monster else 0,
        def_raw=int(def_) if is_monster else 0,
        atk_norm=(int(atk) / 65535.0) if is_monster else 0.0,
        def_norm=(int(def_) / 65535.0) if is_monster else 0.0,
        types=types,
        pendulum_scale_raw=int(lscale_db) if is_pendulum else 0,
    )


def build_static_intrinsic_table(
    embeddings: dict,
    *,
    db_path: Optional[str] = None,
) -> np.ndarray:
    """
    Rows align with EmbeddingCache order: list(embeddings.keys()) row i == emb_idx i.
    Row is static features [:STATIC_INTRINSIC_DIM] for that card in a neutral pose (Hand).
    """
    if db_path is None:
        db_path = os.getenv("YGO_DB_PATH", "ygo-env/assets/locale/en/cards.cdb")

    card_ids = list(embeddings.keys())
    n = len(card_ids)
    table = np.zeros((n, STATIC_INTRINSIC_DIM), dtype=np.float32)

    conn = sqlite3.connect(db_path)
    try:
        for i, cid in enumerate(card_ids):
            row = conn.execute(
                "SELECT type, atk, def, level, race, attribute FROM datas WHERE id = ?",
                (int(cid),),
            ).fetchone()
            if row is None:
                continue
            type_int, atk, def_, level, race_int, attr_int = row
            cr = _datas_row_to_cardrecord(
                str(cid), type_int, atk, def_, level, race_int, attr_int
            )
            _, _, static_full, _ = _encode_cards_vectorized([cr], embeddings)
            table[i, :] = static_full[0, :STATIC_INTRINSIC_DIM].astype(np.float32)
    finally:
        conn.close()

    return table
