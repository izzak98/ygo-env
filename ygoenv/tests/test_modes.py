from ygoenv.modes import GameMode, OpponentMode, make_env_kwargs, resolve_mode_config


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
    assert kw["play_mode"] == "board"
    assert kw["player"] == 0
    assert kw["use_deck_rewards"] is True
    assert kw["max_cards"] == 80
