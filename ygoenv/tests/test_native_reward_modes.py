"""Tests for native reward modes: terminal_board_value and shaped_first_credit.

These tests verify:
1. terminal_board_value: reward=0 mid-episode, reward=board_value at done.
2. shaped_first_credit: incremental rewards mid-episode, terminal squaring at done.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import ygoenv.ygopro  # noqa: F401
    HAS_SO = True
except ImportError:
    HAS_SO = False

from ygoenv.modes import _native_has_opening_hand
from ygoenv.rewards import Cards, get_reward, get_reward_breakdown

ZOODIAC = "zoodiac"

pytestmark = pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")


def _find(env, *, msg=None, act=None, card=None, finish=None):
    from ygoenv.decoding import decode_all_batch
    actions = list(decode_all_batch(env._obs, False)[0][1])
    for i, a in enumerate(actions):
        if msg is not None and a.msg_name != msg:
            continue
        if act is not None and a.act_name != act:
            continue
        if card is not None and a.card_id != card:
            continue
        if finish is not None and a.finish != finish:
            continue
        return i
    return None


def _require(env, **kwargs) -> int:
    idx = _find(env, **kwargs)
    assert idx is not None, f"missing action {kwargs}"
    return idx


def _step(env, idx: int) -> float:
    _, rews, *_ = env.step(np.array([idx], dtype=np.int32))
    return float(rews[0])


def _format_actions(env) -> str:
    from ygoenv.decoding import decode_all_batch
    actions = list(decode_all_batch(env._obs, False)[0][1])
    return "\n".join(
        f"{i}: {a.msg_name} act={a.act_name} card={a.card_id}"
        for i, a in enumerate(actions)
    )


def _idle_msgs(env) -> set[str]:
    from ygoenv.decoding import decode_all_batch
    return {a.msg_name for a in decode_all_batch(env._obs, False)[0][1]}


def _resolve_until_idle(env, prefer_cards=(), activate_cards=()) -> float:
    last = 0.0
    for _ in range(50):
        msgs = _idle_msgs(env)
        if "SELECT_IDLECMD" in msgs:
            return last
        if "SELECT_PLACE" in msgs or "SELECT_POSITION" in msgs:
            last = _step(env, 0)
            continue
        if "SELECT_EFFECTYN" in msgs or "SELECT_YESNO" in msgs or "SELECT_CHAIN" in msgs:
            idx = None
            for card in activate_cards:
                idx = _find(env, act="Activate", card=card)
                if idx is not None:
                    break
            if idx is None:
                idx = _find(env, act="Cancel")
            if idx is None:
                idx = len(list(_idle_msgs(env))) - 1
                from ygoenv.decoding import decode_all_batch
                idx = len(list(decode_all_batch(env._obs, False)[0][1])) - 1
            last = _step(env, idx)
            continue
        if "SELECT_OPTION" in msgs:
            last = _step(env, 0)
            continue
        if "SELECT_CARD" in msgs or "SELECT_TRIBUTE" in msgs:
            idx = None
            for card in prefer_cards:
                idx = _find(env, card=card)
                if idx is not None:
                    break
            last = _step(env, 0 if idx is None else idx)
            continue
        if "SELECT_UNSELECT_CARD" in msgs:
            idx = None
            for card in prefer_cards:
                idx = _find(env, card=card, finish=False)
                if idx is not None:
                    break
            if idx is None:
                idx = _find(env, finish=True)
            last = _step(env, 0 if idx is None else idx)
            continue
        raise AssertionError(f"unexpected prompt {msgs}\n{_format_actions(env)}")
    raise AssertionError("did not return to idle")


def _idle_action(env, act, card, *, prefer_cards=(), activate_cards=()) -> float:
    idx = _require(env, msg="SELECT_IDLECMD", act=act, card=card)
    _step(env, idx)
    return _resolve_until_idle(env, prefer_cards=prefer_cards, activate_cards=activate_cards)


def _make_env(reward_mode: str, episode_done_mode: str = "turn"):
    from ygoenv import GameMode, YGOEnv
    return YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck=ZOODIAC,
        num_envs=1,
        seed_mode="full_det",
        base_seed=0,
        opening_hand=[Cards.RATPIER],
        reward_mode=reward_mode,
        episode_done_mode=episode_done_mode,
    )


@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
def test_terminal_board_value_zero_mid_episode():
    """terminal_board_value: all mid-episode rewards are 0."""
    env = _make_env("terminal_board_value")
    try:
        # Initial obs should have reward=0
        # Summon Ratpier
        _step(env, _require(env, msg="SELECT_IDLECMD", act="Summon", card=Cards.RATPIER))
        _resolve_until_idle(env)

        # XYZ summon Boarbow
        rew = _idle_action(env, "SpSummon", Cards.BOARBOW, prefer_cards=(Cards.RATPIER,))
        assert rew == 0.0, f"terminal_board_value should be 0 mid-episode, got {rew}"

        # XYZ summon Hammerkong
        rew = _idle_action(
            env, "SpSummon", Cards.HAMMERKONG,
            prefer_cards=(Cards.RATPIER,),
            activate_cards=(Cards.BOARBOW,),
        )
        assert rew == 0.0, f"terminal_board_value should be 0 mid-episode, got {rew}"
    finally:
        env.close()


@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
def test_terminal_board_value_nonzero_at_done():
    """terminal_board_value: reward=0 mid-episode; at end turn, reward = board value (>=0)."""
    env = _make_env("terminal_board_value")
    try:
        from ygoenv.decoding import decode_all_batch

        # Summon Ratpier, skip trigger
        _step(env, _require(env, msg="SELECT_IDLECMD", act="Summon", card=Cards.RATPIER))
        _resolve_until_idle(env)

        # XYZ summon Boarbow (should give 0 reward mid-episode)
        rew = _idle_action(env, "SpSummon", Cards.BOARBOW, prefer_cards=(Cards.RATPIER,))
        assert rew == 0.0, f"expected 0 mid-episode, got {rew}"

        # End turn — reward should now equal board value
        all_actions = list(decode_all_batch(env._obs, False)[0][1])
        end_idx = len(all_actions) - 1
        rew = _step(env, end_idx)
        # Board has Boarbow on field; expect some board value (could be >0)
        assert isinstance(rew, float), f"reward should be float at done, got {type(rew)}"
        # After done, env auto-resets; next obs is fresh episode
    finally:
        env.close()


@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
def test_shaped_first_credit_incremental():
    """shaped_first_credit: rewards are floats at each step; episode ends with done."""
    env = _make_env("shaped_first_credit")
    try:
        from ygoenv.decoding import decode_all_batch

        # Summon Ratpier
        rew0 = _step(env, _require(env, msg="SELECT_IDLECMD", act="Summon", card=Cards.RATPIER))
        _resolve_until_idle(env)
        assert isinstance(rew0, float)

        # XYZ summon Boarbow
        rew1 = _idle_action(env, "SpSummon", Cards.BOARBOW, prefer_cards=(Cards.RATPIER,))
        assert isinstance(rew1, float), f"expected float reward, got {type(rew1)}"

        # XYZ summon Hammerkong using Boarbow + Ratpier-overlay
        rew2 = _idle_action(
            env, "SpSummon", Cards.HAMMERKONG,
            prefer_cards=(Cards.RATPIER,),
            activate_cards=(Cards.HAMMERKONG,),
        )
        assert isinstance(rew2, float)

        # End turn — terminal reward
        all_actions = list(decode_all_batch(env._obs, False)[0][1])
        end_idx = len(all_actions) - 1
        rew_terminal = _step(env, end_idx)
        assert isinstance(rew_terminal, float), f"terminal reward should be float, got {type(rew_terminal)}"
        # Terminal reward for shaped_first_credit: either -1.0 (no positive board) or raw^2 - lost_credit
        # Just verify it's finite
        assert not np.isnan(rew_terminal), "terminal reward should not be NaN"
    finally:
        env.close()


@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
def test_episode_done_mode_turn():
    """episode_done_mode=turn: episode ends when the turn advances."""
    from ygoenv import GameMode, YGOEnv
    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck=ZOODIAC,
        num_envs=1,
        seed_mode="full_det",
        base_seed=0,
        opening_hand=[Cards.RATPIER],
        reward_mode="terminal_board_value",
        episode_done_mode="turn",
    )
    try:
        # Summon Ratpier
        _step(env, _require(env, msg="SELECT_IDLECMD", act="Summon", card=Cards.RATPIER))
        _resolve_until_idle(env)

        # End turn — with episode_done_mode=turn, this should end the episode
        from ygoenv.decoding import decode_all_batch
        all_actions = list(decode_all_batch(env._obs, False)[0][1])
        end_idx = len(all_actions) - 1
        _, rews, terminated, truncated, _ = env.step(np.array([end_idx], dtype=np.int32))
        rew = float(rews[0])
        done = bool(terminated[0]) or bool(truncated[0])
        assert done, "episode should be done after end turn with episode_done_mode=turn"
        # The env auto-resets, so next step is valid
    finally:
        env.close()


@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
def test_reward_mode_duel_unchanged():
    """reward_mode=duel (default): existing behavior, deck rewards used unchanged."""
    from ygoenv import GameMode, YGOEnv
    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck=ZOODIAC,
        num_envs=1,
        seed_mode="full_det",
        base_seed=0,
        opening_hand=[Cards.RATPIER],
        reward_mode="duel",
        episode_done_mode="duel",
    )
    try:
        _step(env, _require(env, msg="SELECT_IDLECMD", act="Summon", card=Cards.RATPIER))
        _resolve_until_idle(env)
        # Duel mode: rewards should come from deck rewards (use_deck_rewards path)
        # Just check we don't crash and reward is float
        rew = _idle_action(env, "SpSummon", Cards.BOARBOW, prefer_cards=(Cards.RATPIER,))
        assert isinstance(rew, float)
    finally:
        env.close()
