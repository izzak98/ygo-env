"""High-level game modes mapped to low-level YGOPro env config."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict


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
    use_deck_rewards: bool
    greedy_reward: bool


def resolve_mode_config(
    mode: GameMode,
    *,
    opponent_mode: OpponentMode = OpponentMode.BOT,
    greedy_reward: bool = True,
    ai_player: int = 0,
) -> ModeConfig:
    """Map a :class:`GameMode` to engine configuration."""
    if mode == GameMode.BOARD_SETUP:
        return ModeConfig(
            play_mode="board",
            player=0,
            use_deck_rewards=True,
            greedy_reward=False,
        )
    if mode == GameMode.PLAY_VS_OPPONENT:
        return ModeConfig(
            play_mode=opponent_mode.value,
            player=ai_player,
            use_deck_rewards=False,
            greedy_reward=greedy_reward,
        )
    raise ValueError(f"Unknown game mode: {mode!r}")


@lru_cache(maxsize=1)
def _legacy_native_build() -> bool:
    """True when the installed ``ygopro_ygoenv.so`` predates script/reward JSON wiring."""
    try:
        from ygoenv.ygopro import init_module

        return len(inspect.signature(init_module).parameters) <= 3
    except Exception:
        return True


def make_env_kwargs(
    mode: GameMode,
    *,
    deck1: str,
    deck2: str,
    max_cards: int,
    opponent_mode: OpponentMode = OpponentMode.BOT,
    greedy_reward: bool = True,
    ai_player: int = 0,
    verbose: bool = False,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Build the ``ygoenv.make()`` keyword dict for *mode*."""
    cfg = resolve_mode_config(
        mode,
        opponent_mode=opponent_mode,
        greedy_reward=greedy_reward,
        ai_player=ai_player,
    )
    play_mode = cfg.play_mode
    if _legacy_native_build() and play_mode == "board":
        # Older ygopro_ygoenv.so builds lack board-setup play mode.
        play_mode = "bot"
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
        "play_mode": play_mode,
        "async_reset": False,
        "verbose": verbose,
        "record": False,
        "oppo_info": True,
        "greedy_reward": cfg.greedy_reward,
    }
    if not _legacy_native_build():
        out["use_deck_rewards"] = cfg.use_deck_rewards
    if seed is not None:
        out["seed"] = seed
    return out
