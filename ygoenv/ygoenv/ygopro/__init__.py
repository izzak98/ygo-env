from ygoenv.python.api import py_env

from .ygopro_ygoenv import (
    _YGOProEnvPool,
    _YGOProEnvSpec,
    init_module,
)

try:
    from .ygopro_ygoenv import load_reward_json
except ImportError:
    def load_reward_json(reward_json: str) -> None:  # noqa: ARG001
        """No-op when native binding was built without load_reward_json."""
        pass

try:
    from .ygopro_ygoenv import set_emb_index_map
except ImportError:
    def set_emb_index_map(code_to_emb_idx) -> None:  # noqa: ARG001
        """No-op when native binding was built without set_emb_index_map."""
        pass

(
    YGOProEnvSpec,
    YGOProDMEnvPool,
    YGOProGymEnvPool,
    YGOProGymnasiumEnvPool,
) = py_env(_YGOProEnvSpec, _YGOProEnvPool)


__all__ = [
    "YGOProEnvSpec",
    "YGOProDMEnvPool",
    "YGOProGymEnvPool",
    "YGOProGymnasiumEnvPool",
    "init_module",
    "load_reward_json",
    "set_emb_index_map",
]
