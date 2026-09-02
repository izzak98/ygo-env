"""Observation encoding and action-space constants shared by env_wrapping and training code."""

from __future__ import annotations

import os
import warnings

if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    warnings.warn("No CUDA device found, using CPU. This will be slow.")

LONG_PROMPTS_TO_SHORT_PROMPTS: dict[str, str] = {
    "SELECT_IDLECMD": "idle_cmnd",
    "SELECT_CHAIN": "chain",
    "SELECT_PLACE": "placement",
    "SELECT_CARD": "select_card",
    "SELECT_UNSELECT_CARD": "unselect_card",
    "SELECT_POSITION": "position",
    "SELECT_OPTION": "option",
    "SELECT_TRIBUTE": "tribute",
    "SELECT_SUM": "sum",
    "SELECT_DISFIELD": "disfield",
    "SELECT_ANNOUNCE_ATTRIBUTE": "announce_attribute",
    "SELECT_ANNOUNCE_NUMBER": "announce_number",
    "SELECT_BINARY": "binary",
    "SELECT_EFFECTYN": "binary",
    "SELECT_YESNO": "binary",
    "ANNOUNCE_ATTRIB": "announce_attribute",
    "ANNOUNCE_NUMBER": "announce_number",
}

PROMPTS = [
    "idle_cmnd",
    "chain",
    "placement",
    "select_card",
    "unselect_card",
    "position",
    "binary",
    "option",
    "tribute",
    "sum",
    "disfield",
    "announce_attribute",
    "announce_number",
]

EMBEDDING_SIZE = 1024
COMMAND_SIZE = 11
ACT_SIZE = 8
ACTION_SIZE = COMMAND_SIZE + ACT_SIZE  # 20
CARD_ENCODING_SIZE = 83
CARD_FEATURE_SIZE = EMBEDDING_SIZE + CARD_ENCODING_SIZE + ACTION_SIZE  # 1127
HISTORY_FEATURE_SIZE = EMBEDDING_SIZE + 23  # 1047

STATIC_CARD_DIM = 85
STATIC_INTRINSIC_DIM = 68
STATIC_BOARD_DIM = STATIC_CARD_DIM - STATIC_INTRINSIC_DIM  # 17
DYNAMIC_CARD_DIM = 7 + ACTION_SIZE  # Location (7) + actionability (20) = 27
HISTORY_INFO_DIM = 23  # Command types (11) + action types (8) + padding = 23
N_CARDS = 80

BINARY_SIZE = 2
CARD_SIZE = 80
EFFECT_SIZE = 8
OPTION_SIZE = 30
PLACEMENT_SIZE = 7
POSITION_SIZE = 7
PROMPT_SIZE = len(PROMPTS)

BINARY_START = 0
CARD_START = BINARY_START + BINARY_SIZE
EFFECT_START = CARD_START + CARD_SIZE
OPTION_START = EFFECT_START + EFFECT_SIZE
PLACEMENT_START = OPTION_START + OPTION_SIZE
POSITION_START = PLACEMENT_START + PLACEMENT_SIZE
PROMPT_START = POSITION_START + POSITION_SIZE
N_TOKENS = PROMPT_START + PROMPT_SIZE

CMND_INDICES = {
    "SELECT_IDLECMD": 0,
    "SELECT_CHAIN": 1,
    "SELECT_CARD": 2,
    "SELECT_TRIBUTE": 3,
    "SELECT_POSITION": 4,
    "SELECT_EFFECTYN": 5,
    "SELECT_YESNO": 6,
    "SELECT_UNSELECT_CARD": 7,
    "SELECT_OPTION": 8,
    "SELECT_PLACE": 9,
    "SELECT_SUM": 10,
}

ACT_INDICES = {
    "Set": 11,
    "Repos": 12,
    "SpSummon": 13,
    "Summon": 14,
    "MSet": 15,
    "Attack": 16,
    "DirectAttack": 17,
    "Activate": 18,
}

LINK_MARKER_BIT_TO_ARROW: dict[int, str] = {
    0x001: "bottom_left",
    0x002: "bottom",
    0x004: "bottom_right",
    0x008: "left",
    0x020: "right",
    0x040: "top_left",
    0x080: "top",
    0x100: "top_right",
}

LINK_ARROW_BITS: list[int] = [
    0x001,
    0x002,
    0x004,
    0x008,
    0x020,
    0x040,
    0x080,
    0x100,
]

_LINK_MARKER_MASK = 0x1FF

LINK_DEFENSE_TO_ARROWS: dict[int, list[str]] = {}
for _v in range(_LINK_MARKER_MASK + 1):
    _arrows: list[str] = []
    for _bit, _arrow_name in sorted(LINK_MARKER_BIT_TO_ARROW.items()):
        if _v & _bit:
            _arrows.append(_arrow_name)
    LINK_DEFENSE_TO_ARROWS[_v] = _arrows


def decode_link_arrows(defense: int) -> list[str]:
    return LINK_DEFENSE_TO_ARROWS.get(defense & _LINK_MARKER_MASK, [])


IDLE_PROMPT_IDX = PROMPTS.index("idle_cmnd")
CHAIN_PROMPT_IDX = PROMPTS.index("chain")

PROMPT_TOKEN_SECTION: dict[str, tuple[int, int]] = {
    "binary":             (BINARY_START,    BINARY_SIZE),
    "placement":          (PLACEMENT_START, PLACEMENT_SIZE),
    "position":           (POSITION_START,  POSITION_SIZE),
    "option":             (OPTION_START,    OPTION_SIZE),
    "disfield":           (OPTION_START,    OPTION_SIZE),
    "announce_attribute": (OPTION_START,    OPTION_SIZE),
    "announce_number":    (OPTION_START,    OPTION_SIZE),
    "select_card":        (CARD_START,      CARD_SIZE),
    "unselect_card":      (CARD_START,      CARD_SIZE),
    "tribute":            (CARD_START,      CARD_SIZE),
    "sum":                (CARD_START,      CARD_SIZE),
}
