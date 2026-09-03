"""One-Ratpier Zoodiac line: legal actions at each idle and max board reward."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import ygoenv.ygopro  # noqa: F401
    HAS_SO = True
except ImportError:
    HAS_SO = False

from ygoenv.decoding import decode_all_batch
from ygoenv.modes import _native_has_opening_hand
from ygoenv.rewards import Cards, get_reward, get_reward_breakdown

DECK = "zoodiac"
DRACO_RULE = "F0 Utopic Draco Future on field (not as material)"
DRIDENT_RULE = "Drident on field (not as material)"


def _actions(env):
    return list(decode_all_batch(env._obs, False)[0][1])


def _format_actions(env) -> str:
    lines = []
    for i, a in enumerate(_actions(env)):
        lines.append(
            f"{i}: {a.msg_name} act={a.act_name} card={a.card_id} eff={a.effect_id} fin={a.finish}"
        )
    return "\n".join(lines)


def _find(env, *, msg=None, act=None, card=None, finish=None):
    for i, a in enumerate(_actions(env)):
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
    assert idx is not None, f"missing action {kwargs}\n{_format_actions(env)}"
    return idx


def _step(env, idx: int) -> float:
    _, rews, *_ = env.step(np.array([idx], dtype=np.int32))
    return float(rews[0])


def _idle_msgs(env) -> set[str]:
    return {a.msg_name for a in _actions(env)}


def _resolve_until_idle(env, prefer_cards=()) -> float:
    total = 0.0
    for _ in range(40):
        msgs = _idle_msgs(env)
        if "SELECT_IDLECMD" in msgs:
            return total
        if "SELECT_PLACE" in msgs or "SELECT_POSITION" in msgs:
            total += _step(env, 0)
            continue
        if "SELECT_EFFECTYN" in msgs or "SELECT_YESNO" in msgs or "SELECT_CHAIN" in msgs:
            idx = _find(env, act="Cancel")
            if idx is None:
                idx = len(_actions(env)) - 1
            total += _step(env, idx)
            continue
        if "SELECT_CARD" in msgs or "SELECT_TRIBUTE" in msgs:
            idx = None
            for card in prefer_cards:
                idx = _find(env, card=card)
                if idx is not None:
                    break
            total += _step(env, 0 if idx is None else idx)
            continue
        if "SELECT_UNSELECT_CARD" in msgs:
            idx = None
            for card in prefer_cards:
                idx = _find(env, card=card, finish=False)
                if idx is not None:
                    break
            if idx is None:
                idx = _find(env, finish=True)
            total += _step(env, 0 if idx is None else idx)
            continue
        raise AssertionError(f"unexpected prompt {msgs}\n{_format_actions(env)}")
    raise AssertionError(f"did not return to idle\n{_format_actions(env)}")


def _resolve_place_pos(env) -> float:
    last = 0.0
    for _ in range(10):
        msgs = _idle_msgs(env)
        if "SELECT_PLACE" in msgs or "SELECT_POSITION" in msgs:
            last = _step(env, 0)
            continue
        return last
    raise AssertionError(f"still placing\n{_format_actions(env)}")


def _idle_action(env, act: str, card: str, *, prefer_cards=()) -> float:
    idx = _require(env, msg="SELECT_IDLECMD", act=act, card=card)
    r0 = _step(env, idx)
    r1 = _resolve_until_idle(env, prefer_cards=prefer_cards)
    return r0 + r1


def _py_reward(env):
    cards = list(env.decoded_cards[0])
    return get_reward(DECK, cards), get_reward_breakdown(DECK, cards)


def _on_field(env, card_id: str, *, overlay: bool = False) -> bool:
    return any(
        c.card_id == card_id
        and c.owner == "me"
        and c.location == "Main Monster Zone"
        and c.overlay is overlay
        for c in env.decoded_cards[0]
    )


@pytest.mark.skipif(not HAS_SO, reason="ygopro_ygoenv.so not built")
@pytest.mark.skipif(not _native_has_opening_hand(), reason="native opening_hand not built")
def test_zoodiac_one_ratpier_max_reward_line():
    from ygoenv import GameMode, YGOEnv

    env = YGOEnv(
        mode=GameMode.BOARD_SETUP,
        deck=DECK,
        num_envs=1,
        seed_mode="full_det",
        base_seed=0,
        opening_hand=[Cards.RATPIER],
        reward_mode="shaped_first_credit",
        episode_done_mode="turn",
    )
    try:
        assert _find(env, msg="SELECT_IDLECMD", act="Summon", card=Cards.RATPIER) is not None
        py, br = _py_reward(env)
        assert py == 0.0 and br == {}

        _step(env, _require(env, msg="SELECT_IDLECMD", act="Summon", card=Cards.RATPIER))
        _resolve_place_pos(env)
        yn = _find(env, msg="SELECT_EFFECTYN", act="Activate", card=Cards.RATPIER)
        assert yn is not None, f"Ratpier on-summon trigger missing\n{_format_actions(env)}"
        _step(env, _require(env, msg="SELECT_EFFECTYN", act="Cancel"))
        _resolve_until_idle(env)
        assert _on_field(env, Cards.RATPIER)
        py, br = _py_reward(env)
        assert py == 0.0 and br == {}

        engine = _idle_action(env, "SpSummon", Cards.BOARBOW, prefer_cards=(Cards.RATPIER,))
        assert _on_field(env, Cards.BOARBOW)
        assert _on_field(env, Cards.RATPIER, overlay=True)
        assert engine == 0.0
        py, br = _py_reward(env)
        assert py == 0.0 and br == {}

        # Granted "detach, SS Ratpier" is attached to Ratpier as material, not Boarbow.
        engine = _idle_action(env, "Activate", Cards.RATPIER, prefer_cards=(Cards.RATPIER,))
        assert _on_field(env, Cards.BOARBOW)
        assert _on_field(env, Cards.RATPIER)
        assert engine == 0.0

        engine = _idle_action(
            env, "SpSummon", Cards.HAMMERKONG, prefer_cards=(Cards.RATPIER,)
        )
        assert _on_field(env, Cards.HAMMERKONG)
        assert _on_field(env, Cards.RATPIER, overlay=True)
        assert engine == 0.0

        engine = _idle_action(env, "Activate", Cards.RATPIER, prefer_cards=(Cards.RATPIER,))
        assert _on_field(env, Cards.HAMMERKONG)
        assert _on_field(env, Cards.RATPIER)
        assert engine == 0.0

        engine = _idle_action(
            env,
            "SpSummon",
            Cards.F0_UTOPIC_FUTURE,
            prefer_cards=(Cards.BOARBOW, Cards.HAMMERKONG),
        )
        assert _on_field(env, Cards.F0_UTOPIC_FUTURE)
        assert engine == 0.0
        py, br = _py_reward(env)
        assert py == 0.0 and br == {}

        engine = _idle_action(
            env, "SpSummon", Cards.F0_DRACO_FUTURE, prefer_cards=(Cards.F0_UTOPIC_FUTURE,)
        )
        assert _on_field(env, Cards.F0_DRACO_FUTURE)
        py, br = _py_reward(env)
        assert py == pytest.approx(1.0)
        assert br == {DRACO_RULE: 1.0}
        # shaped_first_credit: Draco rule credit fires within this idle-action block
        assert engine == pytest.approx(1.0), f"Draco rule should give +1.0 credit, got {engine}"

        engine = _idle_action(
            env, "SpSummon", Cards.DRIDENT, prefer_cards=(Cards.RATPIER,)
        )
        assert _on_field(env, Cards.DRIDENT)
        assert _on_field(env, Cards.F0_DRACO_FUTURE)
        py, br = _py_reward(env)
        assert py == pytest.approx(2.0)
        assert br == {DRIDENT_RULE: 1.0, DRACO_RULE: 1.0}
        # shaped_first_credit: Drident rule credit fires within this idle-action block
        assert engine == pytest.approx(1.0), f"Drident rule should give +1.0 credit, got {engine}"
    finally:
        env.close()
