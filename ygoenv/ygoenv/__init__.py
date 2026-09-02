# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""EnvPool package for efficient RL environment simulation."""

import ygoenv.entry  # noqa: F401
from ygoenv.registration import (
    list_all_envs,
    make,
    make_dm,
    make_gym,
    make_gymnasium,
    make_spec,
    register,
)


from ygoenv.paths import (
    get_repo_root,
    deck_path,
    cards_db,
    code_list_path,
    scripts_path,
    embeddings_path,
)
from ygoenv.init import init_ygopro
from ygoenv.modes import GameMode, OpponentMode, ModeConfig
from ygoenv.wrapper import YGOEnv
from ygoenv.env_wrapping import EnvWrapper, WrappedObs, ENV_MODES
from ygoenv.env_wrapping.wrapped_obs_utils import (
    encoder_forward_kwargs,
    expand_obs_for_encoder,
    gather_pretrained_embeddings,
    reconstruct_full_card_static,
)

__version__ = "0.8.4"
__all__ = [
    "register",
    "make",
    "make_dm",
    "make_gym",
    "make_gymnasium",
    "make_spec",
    "list_all_envs",
    "get_repo_root",
    "deck_path",
    "cards_db",
    "code_list_path",
    "scripts_path",
    "embeddings_path",
    "init_ygopro",
    "GameMode",
    "OpponentMode",
    "ModeConfig",
    "YGOEnv",
    "EnvWrapper",
    "WrappedObs",
    "ENV_MODES",
    "encoder_forward_kwargs",
    "expand_obs_for_encoder",
    "gather_pretrained_embeddings",
    "reconstruct_full_card_static",
]