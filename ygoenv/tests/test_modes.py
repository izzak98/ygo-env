from ygoenv.modes import GameMode, OpponentMode, make_env_kwargs, resolve_mode_config


def test_board_setup_config():
    cfg = resolve_mode_config(GameMode.BOARD_SETUP)
    assert cfg.play_mode == "self"
    assert cfg.player == -1
    assert cfg.use_deck_rewards is True


def test_play_vs_opponent_config():
    cfg = resolve_mode_config(GameMode.PLAY_VS_OPPONENT)
    assert cfg.play_mode == "bot"
    assert cfg.use_deck_rewards is False


def test_make_env_kwargs_includes_decks():
    kw = make_env_kwargs(
        GameMode.BOARD_SETUP,
        deck1="tear",
        deck2="garnet",
        max_cards=80,
    )
    assert kw["deck1"] == "tear"
    assert kw["play_mode"] == "self"
    assert kw["max_cards"] == 80
