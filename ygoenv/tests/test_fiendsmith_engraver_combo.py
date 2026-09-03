"""Fiendsmith Engraver lines on Yubel and Live-Twin."""

from __future__ import annotations

import pytest

try:
    import ygoenv.ygopro  # noqa: F401
    HAS_SO = True
except ImportError:
    HAS_SO = False

from combo_helpers import (
    find,
    idle_action,
    in_gy,
    on_field,
    py_reward,
    require,
    spell_on_field,
)
from ygoenv.modes import _native_has_opening_hand
from ygoenv.rewards import Cards

C = Cards

DECKS = ("yubel", "live_twin")

# Seed 8: pinning only Engraver leaves Tract, Lurrie, and Lacrima in the deck
# on both yubel and live_twin, and starts on SELECT_IDLECMD.
BASE_SEED = 8

PATH1_BREAKDOWN = {
    "yubel": {"Ceasar on field": 1.0},
    "live_twin": {
        "Ceaser on field": 1.0,
        "Crimson Lacrima in GY (with desire in extra)": 0.7,
    },
}
PATH2_BREAKDOWN = {
    "yubel": {"Desirae on field (with sequence or requiem in spell/trap zone)": 1.0},
    "live_twin": {
        "Desire on field (with req of either seq or requiem in spell/trap zone)": 1.0
    },
}


def _env(deck: str):
    from ygoenv import GameMode, YGOEnv

    return YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck=deck,
        num_envs=1,
        seed_mode="full_det",
        base_seed=BASE_SEED,
        opening_hand=[C.ENGRAVER],
        reward_mode="shaped_first_credit",
        episode_done_mode="turn",
    )


def _assert_reward(env, deck: str, breakdown: dict[str, float], engine: float | None = None):
    py, br = py_reward(env, deck)
    expected = sum(breakdown.values())
    assert br == breakdown
    assert py == pytest.approx(expected)
    # engine is an incremental shaped_first_credit reward — just verify it's a finite float
    if engine is not None:
        assert isinstance(engine, float) and not __import__("math").isnan(engine)


def _shared_prefix(env):
    require(env, msg="SELECT_IDLECMD", act="Activate", card=C.ENGRAVER)
    idle_action(env, "Activate", C.ENGRAVER, prefer_cards=(C.TRACT,))
    assert find(env, msg="SELECT_IDLECMD", act="Activate", card=C.TRACT) is not None
    idle_action(
        env, "Activate", C.TRACT, prefer_cards=(C.LURRIE,), activate_cards=(C.LURRIE,)
    )
    assert on_field(env, C.LURRIE)
    idle_action(env, "SpSummon", C.REQUIM, prefer_cards=(C.LURRIE,))
    assert on_field(env, C.REQUIM)
    idle_action(
        env,
        "Activate",
        C.REQUIM,
        prefer_cards=(C.CRIMSON_LACRIMA, C.ENGRAVER),
        activate_cards=(C.CRIMSON_LACRIMA,),
    )
    assert on_field(env, C.CRIMSON_LACRIMA)
    assert in_gy(env, C.ENGRAVER)
    # Hand discard + Lacrima mill.
    gy_engravers = [
        c
        for c in env.decoded_cards[0]
        if c.card_id == C.ENGRAVER and c.location == "Graveyard" and c.owner == "me"
    ]
    assert len(gy_engravers) >= 2


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
@pytest.mark.parametrize("deck", DECKS)
def test_fiendsmith_engraver_path_caesar(deck):
    env = _env(deck)
    try:
        _shared_prefix(env)
        require(env, msg="SELECT_IDLECMD", act="Activate", card=C.REQUIM)
        idle_action(env, "Activate", C.REQUIM, prefer_cards=(C.CRIMSON_LACRIMA,))
        assert spell_on_field(env, C.REQUIM)
        idle_action(
            env, "SpSummon", C.NECROQUIP, prefer_cards=(C.CRIMSON_LACRIMA, C.REQUIM)
        )
        assert on_field(env, C.NECROQUIP)
        require(env, msg="SELECT_IDLECMD", act="Activate", card=C.ENGRAVER)
        idle_action(env, "Activate", C.ENGRAVER, prefer_cards=(C.LURRIE,))
        assert on_field(env, C.ENGRAVER)
        engine = idle_action(
            env, "SpSummon", C.CAESAR, prefer_cards=(C.ENGRAVER, C.NECROQUIP)
        )
        assert on_field(env, C.CAESAR)
        _assert_reward(env, deck, PATH1_BREAKDOWN[deck], engine=engine)
    finally:
        env.close()


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
@pytest.mark.parametrize("deck", DECKS)
def test_fiendsmith_engraver_path_desirae(deck):
    env = _env(deck)
    try:
        _shared_prefix(env)
        require(env, msg="SELECT_IDLECMD", act="Activate", card=C.ENGRAVER)
        idle_action(env, "Activate", C.ENGRAVER, prefer_cards=(C.LURRIE,))
        assert on_field(env, C.ENGRAVER)
        idle_action(
            env, "SpSummon", C.SEQUENCE, prefer_cards=(C.ENGRAVER, C.CRIMSON_LACRIMA)
        )
        assert on_field(env, C.SEQUENCE)
        idle_action(
            env,
            "Activate",
            C.SEQUENCE,
            prefer_cards=(C.DESIRAE, C.ENGRAVER, C.REQUIM, C.CRIMSON_LACRIMA),
        )
        assert on_field(env, C.DESIRAE)
        engine = idle_action(env, "Activate", C.SEQUENCE, prefer_cards=(C.DESIRAE,))
        assert on_field(env, C.DESIRAE)
        assert spell_on_field(env, C.SEQUENCE)
        _assert_reward(env, deck, PATH2_BREAKDOWN[deck], engine=engine)
    finally:
        env.close()
