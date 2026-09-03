"""ML training env wrapper: tensor encoding, reward shaping, episode limits.

Reward shaping and episode termination are now handled natively in C++.
EnvWrapper forwards ``reward_mode`` and ``episode_done_mode`` to the native env
and uses the native reward/done signals directly when those modes are active.
"""

from ygoenv.env_wrapping.env_wrapper import EnvWrapper, WrappedObs, ENV_MODES

from ygoenv.decoding import (
    CardRecord,
    ActionRecord,
    HistoryRecord,
    DoubleSliceableRecord,
    decode_all_batch,
)
from ygoenv.rewards import get_reward, get_reward_breakdown, get_reward_rules, DECK_REWARDS

__all__ = [
    "EnvWrapper",
    "WrappedObs",
    "ENV_MODES",
    "CardRecord",
    "ActionRecord",
    "HistoryRecord",
    "DoubleSliceableRecord",
    "decode_all_batch",
    "get_reward",
    "get_reward_breakdown",
    "get_reward_rules",
    "DECK_REWARDS",
]
