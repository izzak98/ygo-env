import pytest

try:
    import ygoenv.ygopro  # noqa: F401
    HAS_SO = True
except ImportError:
    HAS_SO = False

from ygoenv.modes import _native_has_opening_hand
from ygoenv.rewards import Cards


def _my_hand(env):
    return [
        c
        for c in env.decoded_cards[0]
        if c.location == "Hand" and c.owner == "me"
    ]


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
def test_zoodiac_exactly_one_ratpier():
    from ygoenv import GameMode, YGOEnv

    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck="zoodiac",
        num_envs=1,
        seed_mode="full_dynamic",
        opening_hand=[Cards.RATPIER],
    )
    try:
        for seed in range(5):
            env.reset(seed=seed)
            hand = _my_hand(env)
            ids = [c.card_id for c in hand]
            assert len(hand) == 5
            assert ids.count(Cards.RATPIER) == 1
    finally:
        env.close()


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
def test_zoodiac_two_ratpiers():
    from ygoenv import GameMode, YGOEnv

    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck="zoodiac",
        num_envs=1,
        seed_mode="full_det",
        base_seed=1,
        opening_hand=[Cards.RATPIER, Cards.RATPIER],
    )
    try:
        ids = [c.card_id for c in _my_hand(env)]
        assert len(ids) == 5
        assert ids.count(Cards.RATPIER) == 2
    finally:
        env.close()


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
def test_opening_hand_missing_card_raises():
    from ygoenv import GameMode, YGOEnv

    with pytest.raises(ValueError, match="not in main deck"):
        YGOEnv(
            mode=GameMode.BOARD_SETUP,
            deck="zoodiac",
            num_envs=1,
            opening_hand=[1],
        )
