"""One-time YGOPro module initialization (card DB, decks, scripts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ygoenv.paths import cards_db, code_list_path, deck_path, scripts_path
from ygoenv.rewards.config import export_to_json
from ygoenv.ygopro import init_module


def _extract_deck_name(path: Union[str, Path]) -> str:
    return Path(path).stem


def _resolve_deck_file(deck: Union[str, Path]) -> Path:
    deck_fp = Path(deck)
    if deck_fp.suffix != ".ydk":
        deck_fp = deck_path(str(deck))
    return deck_fp


def init_ygopro(
    deck: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    opponent_deck: Optional[str] = "garnet",
    lang: str = "en",
    code_list: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
    preload_tokens: bool = False,
    return_deck_names: bool = False,
    script_dir: Optional[Union[str, Path]] = None,
) -> Union[str, Tuple[str, List[str]]]:
    """Register card DB and decks with the C++ engine.

    *opponent_deck* may be ``None`` to skip a real P2 deck (board setup uses
    the engine-internal ``_dummy`` filler).
    """
    db = Path(db_path) if db_path is not None else cards_db(lang)
    cl = Path(code_list) if code_list is not None else code_list_path()
    scripts = Path(script_dir) if script_dir is not None else scripts_path()

    decks: Dict[str, str] = {}
    deck_dir: Optional[Path] = None
    deck_name = "random"

    deck_items: List[Union[str, Path]]
    if isinstance(deck, (str, Path)):
        deck_items = [deck]
    else:
        deck_items = list(deck)

    for item in deck_items:
        deck_fp = Path(item)
        if deck_fp.is_dir():
            for f in deck_fp.glob("*.ydk"):
                decks[f.stem] = str(f)
            deck_dir = deck_fp
            deck_name = "random"
        else:
            deck_fp = _resolve_deck_file(deck_fp)
            name = _extract_deck_name(deck_fp)
            decks[name] = str(deck_fp)
            deck_dir = deck_fp.parent
            if len(deck_items) == 1:
                deck_name = name

    if opponent_deck:
        opp_fp = deck_path(opponent_deck)
        decks[_extract_deck_name(opp_fp)] = str(opp_fp)

    if preload_tokens:
        if deck_dir is None:
            raise FileNotFoundError("Cannot preload tokens without a deck directory")
        token_deck = deck_dir / "_tokens.ydk"
        if not token_deck.exists():
            raise FileNotFoundError(f"Token deck not found: {token_deck}")
        decks["_tokens"] = str(token_deck)

    reward_json = json.dumps(export_to_json())
    try:
        init_module(
            str(db),
            str(cl),
            decks,
            str(scripts),
            reward_json,
        )
    except TypeError:
        # Older ygopro_ygoenv.so builds only accept (db, code_list, decks).
        init_module(str(db), str(cl), decks)
        from ygoenv.ygopro import load_reward_json

        load_reward_json(reward_json)

    if return_deck_names:
        names = [n for n in decks.keys() if n != "_tokens"]
        return deck_name, names
    return deck_name
