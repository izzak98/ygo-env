from ygoenv.rewards import get_deck_names, get_reward_rules


def test_deck_names_non_empty():
    names = get_deck_names()
    assert len(names) > 0
    assert "tear" in names


def test_reward_rules_for_tear():
    rules = get_reward_rules("tear")
    assert len(rules) > 0
