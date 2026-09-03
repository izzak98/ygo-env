"""Shared helpers for scripted YGO combo tests."""

from __future__ import annotations

import numpy as np

from ygoenv.decoding import decode_all_batch
from ygoenv.rewards import get_reward, get_reward_breakdown


def actions(env):
    return list(decode_all_batch(env._obs, False)[0][1])


def format_actions(env) -> str:
    lines = []
    for i, a in enumerate(actions(env)):
        lines.append(
            f"{i}: {a.msg_name} act={a.act_name} card={a.card_id} "
            f"eff={a.effect_id} place={a.place} fin={a.finish}"
        )
    return "\n".join(lines)


def find(env, *, msg=None, act=None, card=None, finish=None):
    for i, a in enumerate(actions(env)):
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


def require(env, **kwargs) -> int:
    idx = find(env, **kwargs)
    assert idx is not None, f"missing action {kwargs}\n{format_actions(env)}"
    return idx


def step(env, idx: int) -> float:
    _, rews, *_ = env.step(np.array([idx], dtype=np.int32))
    return float(rews[0])


def idle_msgs(env) -> set[str]:
    return {a.msg_name for a in actions(env)}


def resolve_until_idle(
    env,
    *,
    prefer_cards=(),
    activate_cards=(),
    max_steps: int = 50,
) -> float:
    """Auto-resolve place/position/select/chain until SELECT_IDLECMD.

    ``prefer_cards`` are chosen first on SELECT_CARD / SELECT_UNSELECT_CARD.
    ``activate_cards`` are taken on SELECT_CHAIN / SELECT_EFFECTYN / SELECT_YESNO;
    otherwise those windows are cancelled.
    """
    last = 0.0
    activate_cards = tuple(activate_cards)
    prefer_cards = tuple(prefer_cards)
    for _ in range(max_steps):
        msgs = idle_msgs(env)
        if "SELECT_IDLECMD" in msgs:
            return last
        if "SELECT_PLACE" in msgs or "SELECT_POSITION" in msgs:
            last = step(env, 0)
            continue
        if "SELECT_EFFECTYN" in msgs or "SELECT_YESNO" in msgs or "SELECT_CHAIN" in msgs:
            idx = None
            for card in activate_cards:
                idx = find(env, act="Activate", card=card)
                if idx is not None:
                    break
            if idx is None:
                idx = find(env, act="Cancel")
            if idx is None:
                idx = len(actions(env)) - 1
            last = step(env, idx)
            continue
        if "SELECT_OPTION" in msgs:
            last = step(env, 0)
            continue
        if "SELECT_CARD" in msgs or "SELECT_TRIBUTE" in msgs:
            idx = None
            for card in prefer_cards:
                idx = find(env, card=card)
                if idx is not None:
                    break
            last = step(env, 0 if idx is None else idx)
            continue
        if "SELECT_UNSELECT_CARD" in msgs:
            idx = None
            for card in prefer_cards:
                idx = find(env, card=card, finish=False)
                if idx is not None:
                    break
            if idx is None:
                idx = find(env, finish=True)
            last = step(env, 0 if idx is None else idx)
            continue
        raise AssertionError(f"unexpected prompt {msgs}\n{format_actions(env)}")
    raise AssertionError(f"did not return to idle\n{format_actions(env)}")


def resolve_place_pos(env) -> float:
    last = 0.0
    for _ in range(10):
        msgs = idle_msgs(env)
        if "SELECT_PLACE" in msgs or "SELECT_POSITION" in msgs:
            last = step(env, 0)
            continue
        return last
    raise AssertionError(f"still placing\n{format_actions(env)}")


def idle_action(
    env,
    act: str,
    card: str,
    *,
    prefer_cards=(),
    activate_cards=(),
) -> float:
    idx = require(env, msg="SELECT_IDLECMD", act=act, card=card)
    step(env, idx)
    return resolve_until_idle(
        env, prefer_cards=prefer_cards, activate_cards=activate_cards
    )


def py_reward(env, deck: str):
    cards = list(env.decoded_cards[0])
    return get_reward(deck, cards), get_reward_breakdown(deck, cards)


def cards_at(env, card_id: str, location: str, *, overlay: bool = False, owner: str = "me"):
    return [
        c
        for c in env.decoded_cards[0]
        if c.card_id == card_id
        and c.owner == owner
        and c.location == location
        and c.overlay is overlay
    ]


def on_field(env, card_id: str, *, overlay: bool = False) -> bool:
    return bool(cards_at(env, card_id, "Main Monster Zone", overlay=overlay))


def in_gy(env, card_id: str) -> bool:
    return bool(cards_at(env, card_id, "Graveyard"))


def spell_on_field(env, card_id: str) -> bool:
    return bool(cards_at(env, card_id, "Spell & Trap Zone"))
