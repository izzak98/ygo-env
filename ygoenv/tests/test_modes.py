import pytest

from ygoenv.modes import (
    GameMode,
    OpponentMode,
    make_env_kwargs,
    resolve_mode_config,
    resolve_opening_hand,
)


def test_board_setup_config():
    cfg = resolve_mode_config(GameMode.BOARD_SETUP)
    assert cfg.play_mode == "board"
    assert cfg.player == 0
    assert cfg.use_deck_rewards is True
    assert cfg.greedy_reward is False


def test_play_vs_opponent_config():
    cfg = resolve_mode_config(GameMode.PLAY_VS_OPPONENT)
    assert cfg.play_mode == "bot"
    assert cfg.use_deck_rewards is False


def test_make_env_kwargs_includes_decks():
    kw = make_env_kwargs(
        GameMode.BOARD_SETUP,
        deck1="tear",
        deck2="_dummy",
        max_cards=80,
    )
    assert kw["deck1"] == "tear"
    assert kw["deck2"] == "_dummy"
    assert kw["play_mode"] in ("board", "bot")
    assert kw["player"] == 0
    if "use_deck_rewards" in kw:
        assert kw["use_deck_rewards"] is True
    assert kw["max_cards"] == 80


def test_resolve_opening_hand_codes_and_names():
    from ygoenv.rewards import Cards

    assert resolve_opening_hand([78872731]) == "78872731"
    assert resolve_opening_hand([Cards.RATPIER, Cards.RATPIER]) == "78872731,78872731"
    assert resolve_opening_hand(["Zoodiac Ratpier"]) == "78872731"


def test_validate_opening_hand_rejects_missing():
    from ygoenv.modes import validate_opening_hand
    from ygoenv.paths import deck_path
    from ygoenv.rewards import Cards

    ydk = str(deck_path("zoodiac"))
    validate_opening_hand([Cards.RATPIER], deck_ydk=ydk)
    with pytest.raises(ValueError, match="not in main deck"):
        validate_opening_hand([1], deck_ydk=ydk)


def test_make_env_kwargs_opening_hand():
    from ygoenv.modes import _native_has_opening_hand
    from ygoenv.rewards import Cards

    if not _native_has_opening_hand():
        return
    kw = make_env_kwargs(
        GameMode.BOARD_SETUP,
        deck1="zoodiac",
        deck2="_dummy",
        max_cards=80,
        opening_hand=[Cards.RATPIER],
    )
    assert kw["opening_hand"] == "78872731"
