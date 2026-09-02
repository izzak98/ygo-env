"""Resolve ygo-env asset paths relative to the submodule root (not CWD)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_LANG_SHORT = {
    "english": "en",
    "en": "en",
    "chinese": "zh",
    "zh": "zh",
}


@lru_cache(maxsize=1)
def get_repo_root() -> Path:
    """Return the ``ygo-env`` repository root (contains ``assets/``, ``example/``)."""
    override = os.getenv("YGO_ENV_ROOT")
    if override:
        root = Path(override)
        if (root / "assets" / "deck").is_dir():
            return root
        raise FileNotFoundError(f"YGO_ENV_ROOT is not a valid ygo-env root: {root}")

    here = Path(__file__).resolve()
    # Source layout: ygo-env/ygoenv/ygoenv/paths.py → parents[2] == ygo-env
    for ancestor in here.parents:
        if (ancestor / "assets" / "deck").is_dir() and (ancestor / "example" / "code_list.txt").is_file():
            return ancestor

    # Monorepo: discover ygo-env submodule from cwd or parents
    for base in [Path.cwd(), *here.parents]:
        candidate = base / "ygo-env"
        if (candidate / "assets" / "deck").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate ygo-env assets. Set YGO_ENV_ROOT to the ygo-env submodule path "
        "or run from the training repo with ygo-env/ present."
    )


def cards_db(lang: str = "en") -> Path:
    """Path to the SQLite card database for *lang*."""
    override = os.getenv("YGO_DB_PATH")
    if override:
        return Path(override)
    short = _LANG_SHORT.get(lang, lang)
    return get_repo_root() / "assets" / "locale" / short / "cards.cdb"


def deck_path(name: str) -> Path:
    """Path to a deck ``.ydk`` file by stem name (e.g. ``tear`` → ``tear.ydk``)."""
    p = Path(name)
    if p.suffix == ".ydk":
        return p if p.is_absolute() else get_repo_root() / "assets" / "deck" / p.name
    return get_repo_root() / "assets" / "deck" / f"{name}.ydk"


def code_list_path() -> Path:
    """Path to ``example/code_list.txt``."""
    override = os.getenv("YGO_CODE_LIST")
    if override:
        return Path(override)
    return get_repo_root() / "example" / "code_list.txt"


def scripts_path() -> Path:
    """Directory containing YGOPro Lua scripts."""
    override = os.getenv("YGO_SCRIPT_PATH")
    if override:
        return Path(override)
    return get_repo_root() / "third_party" / "ygopro-scripts"


def embeddings_path() -> Path:
    """Path to ``embeddings.json`` (training-repo root or explicit override)."""
    override = os.getenv("YGO_EMBEDDINGS_PATH")
    if override:
        return Path(override)

    # Monorepo: training repo root is parent of ygo-env/
    ygo_root = get_repo_root()
    for candidate in [ygo_root.parent / "embeddings.json", Path.cwd() / "embeddings.json"]:
        if candidate.is_file():
            return candidate

    return Path.cwd() / "embeddings.json"
