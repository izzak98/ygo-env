"""
ML training env wrapper: tensor observation encoding over native-rewarded envs.

Reward shaping and episode termination are fully handled by the native C++ env
(via ``reward_mode`` and ``episode_done_mode`` config keys).  This wrapper's only
job is to transform raw numpy observations into compact ML tensors (WrappedObs).
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Any, Union, Sequence
from dataclasses import dataclass
import threading

import numpy as np
import torch
from ygoenv.wrapper import YGOEnv, OBS_FORMAT_RAW, OBS_FORMAT_VECTORIZED, OBS_FORMATS, try_set_emb_index_map
from ygoenv.modes import GameMode, _native_has_ml_obs, _validate_reward_mode, _validate_episode_done_mode
from ygoenv.decoding import (
    decode_all_batch,
    CardRecord,
    ActionRecord,
    HistoryRecord,
    DoubleSliceableRecord,
    decode_all_batch,
)
from ygoenv.env_wrapping.interface import (
    load_embeddings, encode_all_batch_fast, EncodedObsCompact, encoded_compact_from_ml_numpy,
)
from ygoenv.constants import PROMPTS, LONG_PROMPTS_TO_SHORT_PROMPTS, DEVICE, CARD_SIZE

ENV_MODE_FULL_DET = "full_det"
ENV_MODE_SEMI_DYNAMIC = "semi_dynamic"
ENV_MODE_FULL_DYNAMIC = "full_dynamic"
ENV_MODES = (ENV_MODE_FULL_DET, ENV_MODE_SEMI_DYNAMIC, ENV_MODE_FULL_DYNAMIC)


def get_prompt_one_hot_from_decoded_actions(decoded_actions: List[List[Any]]) -> torch.Tensor:
    prompt_one_hot = torch.zeros((len(decoded_actions), len(PROMPTS)),
                                 device=DEVICE, dtype=torch.float32)
    for i, actions in enumerate(decoded_actions):
        for action in actions:
            prompt_one_hot[i, PROMPTS.index(LONG_PROMPTS_TO_SHORT_PROMPTS[action.msg_name])] = 1.0
    return prompt_one_hot


@dataclass
class WrappedObs:
    """Compact observation: emb indices + per-slot board static; intrinsic static from lazy table."""

    card_emb_idx: torch.Tensor
    hist_emb_idx: torch.Tensor
    card_static_board: torch.Tensor
    card_dynamic: torch.Tensor
    history_info: torch.Tensor
    prompt_one_hot: torch.Tensor
    decoded_cards: DoubleSliceableRecord
    decoded_actions: DoubleSliceableRecord
    decoded_histories: DoubleSliceableRecord

    def __len__(self) -> int:
        return self.card_emb_idx.size(0)

    def __getitem__(self, idx) -> "WrappedObs":
        if isinstance(idx, torch.Tensor):
            index = idx
        else:
            index = idx

        def _slice_tensor(t: torch.Tensor) -> torch.Tensor:
            return t[index]

        def _slice_record(rec: DoubleSliceableRecord) -> DoubleSliceableRecord:
            if isinstance(index, slice):
                indices = list(range(len(rec)))[index]
            elif isinstance(index, torch.Tensor):
                if index.dtype == torch.bool:
                    indices = index.nonzero(as_tuple=True)[0].tolist()
                else:
                    indices = index.detach().cpu().tolist()
            elif isinstance(index, (list, tuple)):
                indices = list(index)
            else:
                indices = [int(index)]

            return DoubleSliceableRecord([rec[i] for i in indices])

        return WrappedObs(
            card_emb_idx=_slice_tensor(self.card_emb_idx),
            hist_emb_idx=_slice_tensor(self.hist_emb_idx),
            card_static_board=_slice_tensor(self.card_static_board),
            card_dynamic=_slice_tensor(self.card_dynamic),
            history_info=_slice_tensor(self.history_info),
            prompt_one_hot=_slice_tensor(self.prompt_one_hot),
            decoded_cards=_slice_record(self.decoded_cards),
            decoded_actions=_slice_record(self.decoded_actions),
            decoded_histories=_slice_record(self.decoded_histories),
        )


class EnvWrapper:
    """
    Wrapper that holds N separate single-env instances, aggregates obs/rewards/dones,
    and applies ML observation encoding.  All reward shaping and episode termination
    come directly from the native C++ env.
    """

    def __init__(
        self,
        deck: Union[str, List[str]],
        num_envs: int,
        *,
        seed: Optional[int] = None,
        db_path: Optional[str] = None,
        verbose: int = 0,
        mode: str = ENV_MODE_FULL_DYNAMIC,
        base_seed: Optional[int] = None,
        auto_reset: bool = True,
        env_verbose: bool = False,
        game_mode: str = "board_setup",
        obs_format: Optional[str] = None,
        opening_hand: Optional[Sequence[Union[int, str]]] = None,
        reward_mode: str = "shaped_first_credit",
        episode_done_mode: str = "turn",
    ):
        self.deck = deck
        self._total_envs = num_envs
        self.num_envs = num_envs
        self.verbose = verbose
        self._auto_reset = auto_reset
        self.env_verbose = env_verbose
        if self.env_verbose:
            assert self.num_envs == 1, "env_verbose only supports num_envs=1"
        if mode not in ENV_MODES:
            raise ValueError(f"mode must be one of {ENV_MODES}, got {mode!r}")
        self._mode = mode
        self._active_indices: Optional[List[int]] = None
        self._base_seed = (
            int(base_seed)
            if base_seed is not None
            else (int(seed) if seed is not None else 0)
        )

        if game_mode not in (GameMode.BOARD_SETUP.value, GameMode.PLAY_VS_OPPONENT.value):
            raise ValueError(
                f"game_mode must be 'board_setup' or 'play_vs_opponent', got {game_mode!r}"
            )
        self._game_mode = GameMode(game_mode)
        _validate_reward_mode(reward_mode)
        _validate_episode_done_mode(episode_done_mode)
        self._reward_mode = reward_mode
        self._episode_done_mode = episode_done_mode
        self._native_reward = reward_mode in ("terminal_board_value", "shaped_first_credit")

        if obs_format is None:
            obs_format = (
                OBS_FORMAT_VECTORIZED
                if self._game_mode == GameMode.BOARD_SETUP
                else OBS_FORMAT_RAW
            )
        if obs_format not in OBS_FORMATS:
            raise ValueError(f"obs_format must be one of {OBS_FORMATS}, got {obs_format!r}")
        if obs_format == OBS_FORMAT_VECTORIZED and self._game_mode == GameMode.PLAY_VS_OPPONENT:
            raise ValueError("obs_format='vectorized' is not supported for play_vs_opponent yet")
        if obs_format == OBS_FORMAT_VECTORIZED and not _native_has_ml_obs():
            warnings.warn(
                "native compact ML obs is unavailable; EnvWrapper falling back to encode_all_batch_fast",
                stacklevel=2,
            )
            obs_format = OBS_FORMAT_RAW
        self.obs_format = obs_format
        self._native_ml = obs_format == OBS_FORMAT_VECTORIZED
        self._warned_ml_fallback = False

        self.embeddings = load_embeddings()
        if self._native_ml:
            try_set_emb_index_map()

        self._ygo = YGOEnv(
            mode=self._game_mode,
            deck=deck,
            num_envs=num_envs,
            seed_mode=mode,
            base_seed=self._base_seed,
            db_path=db_path,
            max_cards=CARD_SIZE,
            verbose=env_verbose,
            obs_format=self.obs_format,
            opening_hand=opening_hand,
            reward_mode=reward_mode,
            episode_done_mode=episode_done_mode,
        )
        self._envs = self._ygo._envs
        self._obs = self._ygo._obs
        self._executor = self._ygo._executor
        self._wrapped_obs = self.transform_obs(self._obs)

    @property
    def _active_env_indices(self) -> List[int]:
        if self._active_indices is None:
            return list(range(self._total_envs))
        return self._active_indices

    def get_subset(self, decks: List[str]) -> "EnvWrapper":
        if isinstance(self.deck, str):
            per_env_decks = [self.deck] * self._total_envs
        else:
            assert len(self.deck) == self._total_envs
            per_env_decks = self.deck

        selected = [i for i, d in enumerate(per_env_decks) if d in decks]
        if not selected:
            raise ValueError(
                f"get_subset(decks={decks!r}) selected no environments from {per_env_decks!r}"
            )

        self._active_indices = selected
        self.num_envs = len(selected)
        return self

    def transform_obs(self, obs: dict) -> WrappedObs:
        batched_decodes = decode_all_batch(obs, True)
        if self._native_ml and "ml_card_emb_idx_" in obs:
            encoded_out, prompt_one_hot = encoded_compact_from_ml_numpy(obs)
        else:
            if self._native_ml and not self._warned_ml_fallback:
                warnings.warn(
                    "native ML obs keys missing; falling back to encode_all_batch_fast",
                    stacklevel=2,
                )
                self._warned_ml_fallback = True
            encoded_out, prompt_one_hot = encode_all_batch_fast(
                obs,
                self.embeddings,
                max_cards=CARD_SIZE,
                use_preallocated_gpu=False,
            )

        decoded_cards = DoubleSliceableRecord([d[0] for d in batched_decodes])
        decoded_actions = DoubleSliceableRecord([d[1] for d in batched_decodes])
        decoded_histories = DoubleSliceableRecord([d[2] for d in batched_decodes])

        prompt_one_hot = get_prompt_one_hot_from_decoded_actions(
            [d[1] for d in batched_decodes])
        return WrappedObs(
            card_emb_idx=encoded_out.card_emb_idx.clone(),
            hist_emb_idx=encoded_out.hist_emb_idx.clone(),
            card_static_board=encoded_out.card_static_board.clone(),
            card_dynamic=encoded_out.card_dynamic.clone(),
            history_info=encoded_out.history_info.clone(),
            prompt_one_hot=prompt_one_hot,
            decoded_cards=decoded_cards,
            decoded_actions=decoded_actions,
            decoded_histories=decoded_histories,
        )

    def _transform_and_update_wrapped_obs(self, indices: List[int]) -> None:
        if not indices:
            return

        partial_raw = {key: self._obs[key][indices] for key in self._obs}
        full_batch = (len(indices) == self._total_envs)
        batched_decodes = decode_all_batch(partial_raw, True)
        if self._native_ml and "ml_card_emb_idx_" in partial_raw:
            encoded_out, prompt_one_hot = encoded_compact_from_ml_numpy(partial_raw)
        else:
            if self._native_ml and not self._warned_ml_fallback:
                warnings.warn(
                    "native ML obs keys missing; falling back to encode_all_batch_fast",
                    stacklevel=2,
                )
                self._warned_ml_fallback = True
            encoded_out, prompt_one_hot = encode_all_batch_fast(
                partial_raw,
                self.embeddings,
                max_cards=CARD_SIZE,
                use_preallocated_gpu=full_batch,
            )
        decoded_cards_p = DoubleSliceableRecord([d[0] for d in batched_decodes])
        decoded_actions_p = DoubleSliceableRecord([d[1] for d in batched_decodes])
        decoded_histories_p = DoubleSliceableRecord([d[2] for d in batched_decodes])

        prompt_one_hot = get_prompt_one_hot_from_decoded_actions(
            [d[1] for d in batched_decodes])
        partial_wrapped = WrappedObs(
            card_emb_idx=encoded_out.card_emb_idx,
            hist_emb_idx=encoded_out.hist_emb_idx,
            card_static_board=encoded_out.card_static_board,
            card_dynamic=encoded_out.card_dynamic,
            history_info=encoded_out.history_info,
            prompt_one_hot=prompt_one_hot,
            decoded_cards=decoded_cards_p,
            decoded_actions=decoded_actions_p,
            decoded_histories=decoded_histories_p,
        )

        if full_batch:
            self._wrapped_obs = WrappedObs(
                card_emb_idx=partial_wrapped.card_emb_idx.clone(),
                hist_emb_idx=partial_wrapped.hist_emb_idx.clone(),
                card_static_board=partial_wrapped.card_static_board.clone(),
                card_dynamic=partial_wrapped.card_dynamic.clone(),
                history_info=partial_wrapped.history_info.clone(),
                prompt_one_hot=partial_wrapped.prompt_one_hot.clone(),
                decoded_cards=partial_wrapped.decoded_cards,
                decoded_actions=partial_wrapped.decoded_actions,
                decoded_histories=partial_wrapped.decoded_histories,
            )
            return

        idx_t = torch.tensor(indices, dtype=torch.long,
                             device=self._wrapped_obs.card_emb_idx.device)
        for field in (
            "card_emb_idx", "hist_emb_idx", "card_static_board", "card_dynamic",
            "history_info", "prompt_one_hot",
        ):
            getattr(self._wrapped_obs, field)[idx_t] = getattr(partial_wrapped, field).clone()

        for rec_field in ("decoded_cards", "decoded_actions", "decoded_histories"):
            getattr(self._wrapped_obs, rec_field).update_indices(
                indices, getattr(partial_wrapped, rec_field)
            )

    @property
    def obs(self) -> WrappedObs:
        active = self._active_env_indices
        if len(active) == self._total_envs and active == list(range(self._total_envs)):
            return self._wrapped_obs

        import torch as _torch

        idx_tensor = _torch.tensor(active, dtype=_torch.long,
                                   device=self._wrapped_obs.card_emb_idx.device)

        def _slice_tensor(x: _torch.Tensor) -> _torch.Tensor:
            return _torch.index_select(x, 0, idx_tensor)

        return WrappedObs(
            card_emb_idx=_slice_tensor(self._wrapped_obs.card_emb_idx),
            hist_emb_idx=_slice_tensor(self._wrapped_obs.hist_emb_idx),
            card_static_board=_slice_tensor(self._wrapped_obs.card_static_board),
            card_dynamic=_slice_tensor(self._wrapped_obs.card_dynamic),
            history_info=_slice_tensor(self._wrapped_obs.history_info),
            prompt_one_hot=_slice_tensor(self._wrapped_obs.prompt_one_hot),
            decoded_cards=DoubleSliceableRecord(
                [self._wrapped_obs.decoded_cards[i] for i in active]),
            decoded_actions=DoubleSliceableRecord(
                [self._wrapped_obs.decoded_actions[i] for i in active]),
            decoded_histories=DoubleSliceableRecord(
                [self._wrapped_obs.decoded_histories[i] for i in active]),
        )

    def _get_reset_seed(self, env_index: int) -> Optional[int]:
        return self._ygo._get_reset_seed(env_index)

    def set_mode(self, mode: str) -> None:
        if mode not in ENV_MODES:
            raise ValueError(f"mode must be one of {ENV_MODES}, got {mode!r}")
        self._mode = mode
        self._ygo.set_seed_mode(mode)

    def close(self) -> None:
        self._ygo.close()

    def __del__(self) -> None:
        try:
            self._ygo.close()
        except Exception:
            pass

    def get_max_action_idx(self) -> list[int]:
        return [len(self._wrapped_obs.decoded_actions[i]) for i in self._active_env_indices]

    def reset(
        self,
        env_indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> dict:
        if env_indices is None:
            env_indices = list(self._active_env_indices)
        else:
            if self._active_indices is not None:
                active_set = set(self._active_env_indices)
                if not (set(env_indices) <= active_set):
                    env_indices = [self._active_env_indices[i] for i in env_indices]

        lock = threading.Lock()

        def _reset_single_env(i):
            idx = int(i)
            reset_seed = (seed + idx) if seed is not None else self._get_reset_seed(idx)
            done_obs, _ = (
                self._envs[idx].reset(seed=reset_seed)
                if reset_seed is not None
                else self._envs[idx].reset()
            )
            with lock:
                for key in self._obs:
                    if key in done_obs:
                        self._obs[key][idx] = done_obs[key][0]

        if len(env_indices) > 1:
            list(self._executor.map(_reset_single_env, env_indices))
        elif len(env_indices) == 1:
            _reset_single_env(env_indices[0])

        if len(env_indices) == self._total_envs:
            self._wrapped_obs = self.transform_obs(self._obs)
        else:
            self._transform_and_update_wrapped_obs(list(env_indices))
        return self.obs

    @property
    def deck_names(self) -> List[str]:
        return [self.deck[i] for i in self._active_env_indices]

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[WrappedObs, List[float], np.ndarray, np.ndarray, List[float], list[int]]:
        """Step all active envs; use native reward/done signals; re-encode obs.

        Returns:
            obs, rewards, dones_bool, done_idx, raw_rewards, max_action_idx_list
        """
        if isinstance(actions, list):
            actions = np.array(actions, dtype=np.int32)

        active = self._active_env_indices
        if actions.shape[0] != len(active):
            raise ValueError(
                f"actions shape {actions.shape[0]} != active envs {len(active)}"
            )

        step_results: List[Optional[dict]] = [None] * len(active)
        engine_rewards_full = [0.0] * self._total_envs
        native_dones_full = [False] * self._total_envs

        def _step_single_env(i: int, local_idx: int) -> None:
            act = np.array([int(actions[local_idx])], dtype=np.int32)
            ob, rew, terminated, truncated, _ = self._envs[i].step(act)
            step_results[local_idx] = ob
            engine_rewards_full[i] = float(rew[0])
            native_dones_full[i] = bool(terminated[0]) or bool(truncated[0])

        if len(active) > 1:
            list(self._executor.map(
                lambda args: _step_single_env(*args),
                [(env_idx, local_i) for local_i, env_idx in enumerate(active)],
            ))
        else:
            _step_single_env(active[0], 0)

        for local_i, env_idx in enumerate(active):
            ob = step_results[local_i]
            for key in self._obs:
                self._obs[key][env_idx] = ob[key][0]

        self._transform_and_update_wrapped_obs(list(active))

        dones_bool = np.zeros(self._total_envs, dtype=bool)
        for env_idx in active:
            dones_bool[env_idx] = native_dones_full[env_idx]
        done_idx = np.array([i for i in active if native_dones_full[i]], dtype=np.int32)

        rewards = [engine_rewards_full[i] for i in active]

        # For done envs, update obs with the already-reset state (native auto-reset)
        if len(done_idx) > 0:
            pass  # obs for done envs already contain new-episode obs from native auto-reset

        subset_dones_bool = dones_bool[active]
        subset_done_idx = np.array(
            [local_i for local_i, env_idx in enumerate(active) if dones_bool[env_idx]],
            dtype=np.int32,
        )
        return (
            self.obs,
            rewards,
            subset_dones_bool,
            subset_done_idx,
            rewards,
            self.get_max_action_idx(),
        )
