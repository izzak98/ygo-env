"""High-level game modes mapped to low-level YGOPro env config."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from collections.abc import Sequence
from typing import Any, Dict, Optional


class GameMode(str, Enum):
    """User-facing environment mode."""

    BOARD_SETUP = "board_setup"
    PLAY_VS_OPPONENT = "play_vs_opponent"


class OpponentMode(str, Enum):
    """Opponent AI type for :attr:`GameMode.PLAY_VS_OPPONENT`."""

    BOT = "bot"
    RANDOM = "random"


@dataclass(frozen=True)
class ModeConfig:
    """Resolved low-level ``ygoenv.make()`` kwargs fragment for a game mode."""

    play_mode: str
    player: int


def resolve_mode_config(
    mode: GameMode,
    *,
    opponent_mode: OpponentMode = OpponentMode.BOT,
    ai_player: int = 0,
) -> ModeConfig:
    """Map a :class:`GameMode` to engine configuration."""
    if mode == GameMode.BOARD_SETUP:
        return ModeConfig(play_mode="board", player=0)
    if mode == GameMode.PLAY_VS_OPPONENT:
        return ModeConfig(play_mode=opponent_mode.value, player=ai_player)
    raise ValueError(f"Unknown game mode: {mode!r}")


@lru_cache(maxsize=1)
def _native_has_opening_hand() -> bool:
    """True when ygopro_ygoenv.so DefaultConfig includes opening_hand."""
    try:
        from ygoenv.ygopro.ygopro_ygoenv import _YGOProEnvSpec

        return "opening_hand" in list(_YGOProEnvSpec._config_keys)
    except Exception:
        return False


@lru_cache(maxsize=1)
def _native_has_ml_obs() -> bool:
    """True when the native StateSpec includes compact ML observation keys."""
    try:
        from ygoenv.ygopro.ygopro_ygoenv import _YGOProEnvSpec

        return any("ml_card_emb_idx_" in str(k) for k in _YGOProEnvSpec._state_keys)
    except Exception:
        return False


def _lookup_card_code_by_name(name: str, db_path: Optional[str] = None) -> str:
    import sqlite3

    from ygoenv.paths import cards_db

    db = db_path or str(cards_db())
    con = sqlite3.connect(db)
    try:
        exact = con.execute(
            "SELECT id FROM texts WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchall()
        if len(exact) == 1:
            return str(exact[0][0])
        if len(exact) > 1:
            ids = ", ".join(str(r[0]) for r in exact)
            raise ValueError(f"ambiguous card name {name!r} (ids {ids})")
        fuzzy = con.execute(
            "SELECT id, name FROM texts WHERE name LIKE ? COLLATE NOCASE",
            (f"%{name}%",),
        ).fetchall()
        if len(fuzzy) == 1:
            return str(fuzzy[0][0])
        if not fuzzy:
            raise ValueError(f"unknown card name {name!r}")
        matches = ", ".join(f"{n} ({i})" for i, n in fuzzy[:8])
        raise ValueError(f"ambiguous card name {name!r}: {matches}")
    finally:
        con.close()


def resolve_opening_hand(
    cards: Sequence[int | str],
    *,
    db_path: Optional[str] = None,
) -> str:
    """Turn card codes or names into the C++ ``opening_hand`` config string."""
    if not cards:
        return ""
    parts: list[str] = []
    for card in cards:
        if isinstance(card, bool) or not isinstance(card, (int, str)):
            raise TypeError(f"opening_hand entries must be int or str, got {card!r}")
        if isinstance(card, int):
            if card <= 0:
                raise ValueError(f"invalid card code {card}")
            parts.append(str(card))
            continue
        token = card.strip()
        if not token:
            raise ValueError("empty card in opening_hand")
        if token.isdigit():
            parts.append(token)
        else:
            parts.append(_lookup_card_code_by_name(token, db_path=db_path))
    return ",".join(parts)


def main_deck_codes_from_ydk(ydk_path: str) -> list[int]:
    """Return main-deck card codes from a ``.ydk`` file, in file order."""
    codes: list[int] = []
    section: str | None = None
    with open(ydk_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                tag = line[1:].strip().lower()
                if tag == "main":
                    section = "main"
                elif tag == "extra":
                    section = "extra"
                continue
            if line.startswith("!"):
                section = "side"
                continue
            if section == "main" and line.isdigit():
                codes.append(int(line))
    return codes


def validate_opening_hand(
    cards: Sequence[int | str],
    *,
    deck_ydk: str,
    db_path: Optional[str] = None,
) -> str:
    """Resolve ``opening_hand`` and check each copy exists in the main deck."""
    encoded = resolve_opening_hand(cards, db_path=db_path)
    if not encoded:
        return encoded
    from collections import Counter

    want_list = [int(x) for x in encoded.split(",")]
    if len(want_list) > 5:
        raise ValueError(
            f"opening_hand has {len(want_list)} cards but the opening hand is 5"
        )
    have = Counter(main_deck_codes_from_ydk(deck_ydk))
    want = Counter(want_list)
    missing = [
        f"{code} (need {n}, deck has {have[code]})"
        for code, n in want.items()
        if have[code] < n
    ]
    if missing:
        raise ValueError("opening_hand cards not in main deck: " + "; ".join(missing))
    return encoded


_VALID_REWARD_MODES = {"duel", "terminal_board_value", "shaped_first_credit"}
_VALID_EPISODE_DONE_MODES = {"duel", "turn"}


def _validate_reward_mode(mode: str) -> None:
    if mode not in _VALID_REWARD_MODES:
        raise ValueError(
            f"reward_mode must be one of {sorted(_VALID_REWARD_MODES)}, got {mode!r}"
        )


def _validate_episode_done_mode(mode: str) -> None:
    if mode not in _VALID_EPISODE_DONE_MODES:
        raise ValueError(
            f"episode_done_mode must be one of {sorted(_VALID_EPISODE_DONE_MODES)}, got {mode!r}"
        )


def make_env_kwargs(
    mode: GameMode,
    *,
    deck1: str,
    deck2: str,
    max_cards: int,
    opponent_mode: OpponentMode = OpponentMode.BOT,
    ai_player: int = 0,
    verbose: bool = False,
    seed: int | None = None,
    obs_format: str = "raw",
    opening_hand: Sequence[int | str] | None = None,
    db_path: Optional[str] = None,
    reward_mode: str = "duel",
    episode_done_mode: str = "duel",
) -> Dict[str, Any]:
    """Build the ``ygoenv.make()`` keyword dict for *mode*."""
    _validate_reward_mode(reward_mode)
    _validate_episode_done_mode(episode_done_mode)
    cfg = resolve_mode_config(mode, opponent_mode=opponent_mode, ai_player=ai_player)
    if obs_format not in ("raw", "vectorized"):
        raise ValueError(f"obs_format must be 'raw' or 'vectorized', got {obs_format!r}")
    out: Dict[str, Any] = {
        "task_id": "YGOPro-v1",
        "env_type": "gymnasium",
        "num_envs": 1,
        "num_threads": 1,
        "deck1": deck1,
        "deck2": deck2,
        "player": cfg.player,
        "max_cards": max_cards,
        "max_options": 99,
        "n_history_actions": 300,
        "play_mode": cfg.play_mode,
        "async_reset": False,
        "verbose": verbose,
        "record": False,
        "oppo_info": True,
        "obs_format": obs_format,
        "reward_mode": reward_mode,
        "episode_done_mode": episode_done_mode,
    }
    if seed is not None:
        out["seed"] = seed
    if opening_hand:
        if not _native_has_opening_hand():
            raise ValueError(
                "opening_hand requires a ygopro_ygoenv.so build with the opening_hand config key"
            )
        out["opening_hand"] = resolve_opening_hand(opening_hand, db_path=db_path)
    return out
