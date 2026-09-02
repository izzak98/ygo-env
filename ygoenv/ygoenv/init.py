"""One-time YGOPro module initialization (card DB, decks, scripts)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ygoenv.paths import cards_db, code_list_path, deck_path
from ygoenv.ygopro import init_module


def _extract_deck_name(path: Union[str, Path]) -> str:
    return Path(path).stem


def init_ygopro(
    deck: Union[str, Path],
    *,
    opponent_deck: str = "garnet",
    lang: str = "en",
    code_list: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
    preload_tokens: bool = False,
    return_deck_names: bool = False,
) -> Union[str, Tuple[str, List[str]]]:
    """Register card DB and decks with the C++ engine."""
    db = Path(db_path) if db_path is not None else cards_db(lang)
    cl = Path(code_list) if code_list is not None else code_list_path()

    deck_fp = Path(deck)
    if deck_fp.is_dir():
        decks: Dict[str, str] = {f.stem: str(f) for f in deck_fp.glob("*.ydk")}
        deck_dir = deck_fp
        deck_name = "random"
    else:
        if deck_fp.suffix != ".ydk":
            deck_fp = deck_path(str(deck))
        deck_name = _extract_deck_name(deck_fp)
        decks = {deck_name: str(deck_fp)}
        deck_dir = deck_fp.parent

    opp_fp = deck_path(opponent_deck)
    opp_name = _extract_deck_name(opp_fp)
    decks[opp_name] = str(opp_fp)

    if preload_tokens:
        token_deck = deck_dir / "_tokens.ydk"
        if not token_deck.exists():
            raise FileNotFoundError(f"Token deck not found: {token_deck}")
        decks["_tokens"] = str(token_deck)

    init_module(str(db), str(cl), decks)

    if return_deck_names:
        names = list(decks.keys())
        if "_tokens" in names:
            names.remove("_tokens")
        return deck_name, names
    return deck_name
