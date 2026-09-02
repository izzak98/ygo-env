"""
Environment wrapper that owns N separate single-env instances and aggregates
observations, rewards, and dones. All env logic (dones, step limit, resets,
rewards) lives here; callers use wrapper.step(actions) and wrapper.obs.

Modes control seed on reset:
- full_det: one seed for all envs (same hand/trajectory every time).
- semi_dynamic: seed = base_seed + env_idx (deterministic per env, different across envs).
- full_dynamic: no seed (fully random).
"""

import collections
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Any, Union, Dict
from dataclasses import dataclass
import threading

import numpy as np
import torch
from ygoenv.wrapper import YGOEnv
from ygoenv.modes import GameMode
from ygoenv.decoding import (
    decode_cards_batch,
    CardRecord,
    ActionRecord,
    HistoryRecord,
    DoubleSliceableRecord,
    decode_all_batch,
)
from ygoenv.rewards import get_reward, get_reward_breakdown
from ygoenv.env_wrapping.interface import (load_embeddings, encode_all_batch_fast, EncodedObsCompact)
from ygoenv.constants import PROMPTS, LONG_PROMPTS_TO_SHORT_PROMPTS, DEVICE, CARD_SIZE
# Valid mode names for curriculum
ENV_MODE_FULL_DET = "full_det"
ENV_MODE_SEMI_DYNAMIC = "semi_dynamic"
ENV_MODE_FULL_DYNAMIC = "full_dynamic"
ENV_MODES = (ENV_MODE_FULL_DET, ENV_MODE_SEMI_DYNAMIC, ENV_MODE_FULL_DYNAMIC)


def get_prompt_one_hot_from_decoded_actions(decoded_actions: List[List[Any]]) -> torch.Tensor:
    """Get a one-hot prompt from decoded actions."""
    prompt_one_hot = torch.zeros((len(decoded_actions), len(PROMPTS)),
                                 device=DEVICE, dtype=torch.float32)
    for i, actions in enumerate(decoded_actions):
        for action in actions:
            prompt_one_hot[i, PROMPTS.index(LONG_PROMPTS_TO_SHORT_PROMPTS[action.msg_name])] = 1.0
    return prompt_one_hot


def _get_dones(obs: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute done indices and boolean mask from batched obs."""
    b = obs["cards_"].shape[0]
    done_idx = np.array(
        [i for i, d in enumerate(obs["global_"]) if d[4] != 1],
        dtype=np.int32,
    )
    dones = np.zeros(b, dtype=bool)
    if len(done_idx) > 0:
        dones[done_idx] = True
    return done_idx, dones


def _stack_obs(obs_list: List[dict]) -> dict:
    """Stack a list of single-env obs (each batch dim 1) into one batched obs."""
    if not obs_list:
        raise ValueError("obs_list must be non-empty")
    out = {}
    for key in obs_list[0].keys():
        parts = [o[key] for o in obs_list]
        out[key] = np.concatenate(parts, axis=0)
    return out


def _calc_rewards(
    deck: Union[str, List[str]],
    decoded_cards: Any,
    done_idx: np.ndarray,
    claimed_rules_per_env: List[Dict[str, float]],
    termination_idx: Optional[List[int]] = None,
    termination_reward: float = 0.2,
) -> Tuple[List[float], List[float]]:
    """Compute rewards and raw_rewards. Non-done: first-time-only per rule (no re-trigger).

    On done: subtract any per-rule credit that was earned mid-episode but is not
    satisfied in the final state (same rule name absent or value <= 0 in breakdown).
    """
    b = len(decoded_cards)
    rewards = []
    raw_rewards = []
    done_set = set(int(x) for x in done_idx) if len(done_idx) > 0 else set()
    if termination_idx is None:
        termination_idx = []
    for i in range(b):
        if isinstance(deck, list):
            deck_name = deck[i]
        else:
            deck_name = deck
        raw_reward = get_reward(deck_name, decoded_cards[i])
        raw_rewards.append(raw_reward)
        if i in done_set:
            final_breakdown = get_reward_breakdown(deck_name, decoded_cards[i])
            lost_rule_credit = 0.0
            for rule_name, credited in claimed_rules_per_env[i].items():
                if final_breakdown.get(rule_name, 0.0) <= 0.0:
                    lost_rule_credit += credited
            if raw_reward == 0.0:
                reward = -1.0
            else:
                reward = raw_reward
                if i in termination_idx:
                    reward *= (1 + termination_reward)
                reward = reward**2
            reward -= lost_rule_credit
        else:
            breakdown = get_reward_breakdown(deck_name, decoded_cards[i])
            reward = 0.0
            for rule_name, value in breakdown.items():
                if value > 0 and rule_name not in claimed_rules_per_env[i]:
                    reward += value
                    claimed_rules_per_env[i][rule_name] = value
        # reward = np.log(reward + 2)
        rewards.append(reward)
    assert len(rewards) == b
    return rewards, raw_rewards


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
        """
        Enable indexing/slicing over the batch dimension so that WrappedObs
        behaves like a batched tensor container, e.g. obs[idx] or obs[batch_idx].
        """

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
    Wrapper that holds N separate single-env instances (num_envs=1 each),
    aggregates their obs/rewards/dones, and applies step limit and reward logic.
    """

    def __init__(
        self,
        deck: Union[str, List[str]],
        num_envs: int,
        *,
        max_episode_steps: int = 250,
        step_limit_penalty: float = -250.0,
        action_loop_window: int = 20,
        action_loop_penalty: float = -100.0,
        action_loop_enabled: bool = True,
        seed: Optional[int] = None,
        db_path: Optional[str] = None,
        verbose: int = 0,
        mode: str = ENV_MODE_FULL_DYNAMIC,
        base_seed: Optional[int] = None,
        auto_reset: bool = True,
        env_verbose: bool = False,
        game_mode: str = "board_setup",
    ):
        self.deck = deck
        # Total number of envs is fixed at construction time; num_envs may later
        # represent only the number of *active* envs when using get_subset().
        self._total_envs = num_envs
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.step_limit_penalty = step_limit_penalty
        self.action_loop_window = action_loop_window
        self.action_loop_penalty = action_loop_penalty
        self.action_loop_enabled = action_loop_enabled
        self.verbose = verbose
        self._auto_reset = auto_reset
        self.env_verbose = env_verbose
        if self.env_verbose:
            assert self.num_envs == 1, "env_verbose only supports num_envs=1"
        if mode not in ENV_MODES:
            raise ValueError(f"mode must be one of {ENV_MODES}, got {mode!r}")
        self._mode = mode
        # When None, all envs [0, _total_envs) are active. When a list of ints,
        # only those indices are considered active by high-level APIs.
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

        self._ygo = YGOEnv(
            mode=self._game_mode,
            deck=deck,
            num_envs=num_envs,
            seed_mode=mode,
            base_seed=self._base_seed,
            db_path=db_path,
            max_cards=CARD_SIZE,
            verbose=env_verbose,
        )
        self._envs = self._ygo._envs
        self._obs = self._ygo._obs
        self._executor = self._ygo._executor

        self.embeddings = load_embeddings()
        self._wrapped_obs = self.transform_obs(self._obs)
        self._prev_rewards = [0.0] * self._total_envs
        self._claimed_reward_rules = [{} for _ in range(self._total_envs)]
        self._terminated = [False] * self._total_envs
        self._steps_since_reset = [0] * self._total_envs
        if self.action_loop_enabled:
            self._action_history = [
                collections.deque(maxlen=action_loop_window)
                for _ in range(self._total_envs)
            ]
            self._action_hashes_seen = [set() for _ in range(self._total_envs)]
        else:
            self._action_history = []
            self._action_hashes_seen = []

    @property
    def _active_env_indices(self) -> List[int]:
        """
        Return the list of underlying env indices that are currently active.
        When no subset has been requested, this is all envs.
        """
        if self._active_indices is None:
            return list(range(self._total_envs))
        return self._active_indices

    def get_subset(self, decks: List[str]) -> "EnvWrapper":
        """
        Restrict this wrapper to operate only on envs whose deck name is in `decks`.

        This reuses the existing underlying env instances and their current state,
        but mutates `self` so that public APIs (`obs`, `step`, `reset`, etc.)
        see only the selected subset.
        """
        if isinstance(self.deck, str):
            per_env_decks = [self.deck] * self._total_envs
        else:
            assert len(self.deck) == self._total_envs, "deck list length mismatch"
            per_env_decks = self.deck

        selected: List[int] = [
            i for i, d in enumerate(per_env_decks) if d in decks
        ]
        if not selected:
            raise ValueError(
                f"get_subset(decks={decks!r}) selected no environments from {per_env_decks!r}"
            )

        self._active_indices = selected
        self.num_envs = len(selected)
        return self

    def transform_obs(self, obs: dict) -> WrappedObs:
        # Serial decode-then-encode: same as _transform_and_update_wrapped_obs.
        # Parallel was slower due to GIL contention from Phase 3 Python object creation.
        #
        # use_preallocated_gpu=False: avoid writing to the module-level _pinned_buffers pool.
        # _pinned_buffers is also used by _transform_and_update_wrapped_obs (full-batch path)
        # which assigns views directly into _wrapped_obs.  If transform_obs also wrote to
        # _pinned_buffers (e.g. from the evaluator's EnvWrapper), it would silently overwrite
        # rows in the training env's _wrapped_obs.  _partial_gpu is a separate pool that
        # _wrapped_obs never references after the first step, so it is safe to use here.
        batched_decodes = decode_all_batch(obs, True)
        encoded_out, prompt_one_hot = encode_all_batch_fast(
            obs,
            self.embeddings,
            max_cards=CARD_SIZE,
            use_preallocated_gpu=False,
        )
        b = obs["cards_"].shape[0]

        decoded_cards = DoubleSliceableRecord([d[0] for d in batched_decodes])
        decoded_actions = DoubleSliceableRecord([d[1] for d in batched_decodes])
        decoded_histories = DoubleSliceableRecord([d[2] for d in batched_decodes])

        # Build prompt from decoded actions (ground truth) to prevent prompt/action mismatch.
        prompt_from_decoded = get_prompt_one_hot_from_decoded_actions(
            [d[1] for d in batched_decodes])
        prompt_one_hot = prompt_from_decoded
        # IMPORTANT: clone tensor fields so this wrapper owns its storage.
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
        """Re-transform obs for only the given env indices and splice back into self._wrapped_obs.

        This is the hot-path alternative to a full self._wrapped_obs = self.transform_obs(self._obs).
        Instead of decoding/encoding all _total_envs every step, we process only the envs whose
        raw obs actually changed (the active subset, or the reset subset after auto-reset).

        Three concrete wins over the naive full-batch approach:
          1. decode_all_batch / encode_all_batch_fast run on len(indices) rows, not _total_envs.
          2. Auto-reset no longer triggers a second full transform; it only re-transforms done envs.
          3. Inactive envs (when using get_subset) are never processed at all.
        """
        if not indices:
            return

        # Build a partial raw-obs dict for only the requested indices.
        partial_raw = {key: self._obs[key][indices] for key in self._obs}
        # Use preallocated GPU tensors only for the full-batch case to avoid the 235ms
        # GPU allocator stall. For partial batches (a few done envs), allocate fresh tensors
        # — the small size prevents stalls, and fresh tensors avoid source/dest aliasing
        # when splicing back into self._wrapped_obs (which already holds the preallocated refs).
        full_batch = (len(indices) == self._total_envs)
        # Serial decode-then-encode: parallel was slower due to GIL contention from
        # Phase 3 Python object creation holding the GIL for ~275ms per decode call.
        batched_decodes = decode_all_batch(partial_raw, True)
        encoded_out, prompt_one_hot = encode_all_batch_fast(
            partial_raw,
            self.embeddings,
            max_cards=CARD_SIZE,
            use_preallocated_gpu=full_batch,
        )
        decoded_cards_p = DoubleSliceableRecord([d[0] for d in batched_decodes])
        decoded_actions_p = DoubleSliceableRecord([d[1] for d in batched_decodes])
        decoded_histories_p = DoubleSliceableRecord([d[2] for d in batched_decodes])

        # Build prompt from decoded actions (ground truth) to prevent prompt/action mismatch.
        prompt_from_decoded = get_prompt_one_hot_from_decoded_actions(
            [d[1] for d in batched_decodes])
        prompt_one_hot = prompt_from_decoded
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
            # Full-batch: replace self._wrapped_obs directly — no splice needed, avoids
            # the aliasing check since preallocated GPU buffers are both source and destination.
            # Clone tensor fields so this wrapper doesn't alias the global preallocated buffers.
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

        # Partial batch: splice tensor fields back in-place.
        # clone() breaks aliasing when _wrapped_obs was set by transform_obs (which uses
        # _partial_gpu) and the partial encode also wrote to _partial_gpu — source and
        # destination would reference the same pool, triggering PyTorch's aliasing guard.
        # The clone is cheap: only len(indices) rows, not the full batch.
        idx_t = torch.tensor(indices, dtype=torch.long,
                             device=self._wrapped_obs.card_emb_idx.device)
        for field in (
            "card_emb_idx", "hist_emb_idx", "card_static_board", "card_dynamic",
            "history_info", "prompt_one_hot",
        ):
            getattr(self._wrapped_obs, field)[idx_t] = getattr(partial_wrapped, field).clone()

        # Splice decoded record lists back at the right slots — O(len(indices)) in-place.
        for rec_field in ("decoded_cards", "decoded_actions", "decoded_histories"):
            getattr(self._wrapped_obs, rec_field).update_indices(
                indices, getattr(partial_wrapped, rec_field)
            )

    @property
    def obs(self) -> WrappedObs:
        """Current batched observation (read-only), restricted to active envs."""
        active = self._active_env_indices
        # Fast path: all envs active in order, preserve existing behavior.
        if len(active) == self._total_envs and active == list(range(self._total_envs)):
            return self._wrapped_obs

        import torch as _torch

        idx_tensor = _torch.tensor(active, dtype=_torch.long,
                                   device=self._wrapped_obs.card_emb_idx.device)

        def _slice_tensor(x: _torch.Tensor) -> _torch.Tensor:
            return _torch.index_select(x, 0, idx_tensor)

        decoded_cards = DoubleSliceableRecord(
            [self._wrapped_obs.decoded_cards[i] for i in active]
        )
        decoded_actions = DoubleSliceableRecord(
            [self._wrapped_obs.decoded_actions[i] for i in active]
        )
        decoded_histories = DoubleSliceableRecord(
            [self._wrapped_obs.decoded_histories[i] for i in active]
        )

        return WrappedObs(
            card_emb_idx=_slice_tensor(self._wrapped_obs.card_emb_idx),
            hist_emb_idx=_slice_tensor(self._wrapped_obs.hist_emb_idx),
            card_static_board=_slice_tensor(self._wrapped_obs.card_static_board),
            card_dynamic=_slice_tensor(self._wrapped_obs.card_dynamic),
            history_info=_slice_tensor(self._wrapped_obs.history_info),
            prompt_one_hot=_slice_tensor(self._wrapped_obs.prompt_one_hot),
            decoded_cards=decoded_cards,
            decoded_actions=decoded_actions,
            decoded_histories=decoded_histories,
        )

    def _get_reset_seed(self, env_index: int) -> Optional[int]:
        """Return seed for reset of env at env_index, or None for full_dynamic."""
        return self._ygo._get_reset_seed(env_index)

    def set_mode(self, mode: str) -> None:
        """Set env mode (full_det, semi_dynamic, full_dynamic) for curriculum."""
        if mode not in ENV_MODES:
            raise ValueError(f"mode must be one of {ENV_MODES}, got {mode!r}")
        self._mode = mode
        self._ygo.set_seed_mode(mode)

    def close(self) -> None:
        """Shut down the persistent thread pool and all underlying envs.
        Call this when done with the wrapper to avoid thread leaks."""
        self._ygo.close()

    def __del__(self) -> None:
        try:
            self._ygo.close()
        except Exception:
            pass

    def set_terminated(self, env_indices: List[int]) -> None:
        """Mark envs as terminated (e.g. idle_cmnd). Used to build termination_idx in step."""
        for i in env_indices:
            if 0 <= i < self._total_envs:
                self._terminated[i] = True

    def get_max_action_idx(self) -> list[int]:
        max_action_idx_list: List[int] = []
        for idx in self._active_env_indices:
            decoded_actions = self._wrapped_obs.decoded_actions[idx]
            max_action_idx = len(decoded_actions)
            max_action_idx_list.append(max_action_idx)
        return max_action_idx_list

    def reset(
        self,
        env_indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> dict:
        """Reset envs and return updated batched obs. If env_indices is None, reset all.
        If seed is not None, use seed + env_index for each env (for deterministic replays).
        Threaded for speed.
        """
        if env_indices is None:
            # By default, operate on currently active envs.
            env_indices = list(self._active_env_indices)
        else:
            # When a subset is active: if the given indices are already underlying
            # (e.g. from step() auto_reset), use them as-is; otherwise treat as
            # subset-relative and convert to underlying.
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
                self._obs["cards_"][idx] = done_obs["cards_"][0]
                self._obs["actions_"][idx] = done_obs["actions_"][0]
                self._obs["h_actions_"][idx] = done_obs["h_actions_"][0]
                self._obs["global_"][idx] = done_obs["global_"][0]
                self._terminated[idx] = False
                self._steps_since_reset[idx] = 0
                self._claimed_reward_rules[idx].clear()
                if self.action_loop_enabled:
                    self._action_history[idx].clear()
                    self._action_hashes_seen[idx].clear()

        if len(env_indices) > 1:
            list(self._executor.map(_reset_single_env, env_indices))
        elif len(env_indices) == 1:
            _reset_single_env(env_indices[0])
        # Only re-transform the envs that were actually reset, not the full batch.
        # This avoids a second full decode/encode pass on steps where auto_reset fires.
        if len(env_indices) == self._total_envs:
            # Full reset (e.g. initial setup or explicit global reset): rebuild entirely.
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
        termination_idx: Optional[List[int]] = None,
    ) -> Tuple[dict, List[float], np.ndarray, np.ndarray, List[float], list[int]]:
        """
        Step all envs; aggregate obs; compute dones, step limit, rewards; reset done envs.

        Returns:
            obs: Batched observation (updated in-place and returned).
            rewards: List of length num_envs.
            dones_bool: Boolean array of length num_envs.
            done_idx: Indices of envs that ended this step.
        """
        if isinstance(actions, list):
            actions = np.array(actions, dtype=np.int32)

        active = self._active_env_indices
        if actions.shape[0] != len(active):
            raise ValueError(
                f"actions shape {actions.shape[0]} != active envs {len(active)}"
            )

        # Step each env (each has batch size 1). This is the main per-step bottleneck,
        # so we parallelize across envs using threads.

        # See if any envs are terminated
        terminated_idx: List[int] = []
        max_action_idx_list = self.get_max_action_idx()
        for local_i, env_idx in enumerate(active):
            turn_was_ended = actions[local_i] == max_action_idx_list[local_i] - 1
            if turn_was_ended:
                terminated_idx.append(env_idx)
        if terminated_idx:
            self.set_terminated(terminated_idx)

        # Collect stepped obs into a flat list indexed by local position.
        step_results: List[Optional[dict]] = [None] * len(active)
        engine_rewards_full = [0.0] * self._total_envs

        def _step_single_env(i: int, local_idx: int) -> None:
            act = np.array([int(actions[local_idx])], dtype=np.int32)
            ob, rew, _, _, _ = self._envs[i].step(act)
            step_results[local_idx] = ob
            engine_rewards_full[i] = float(rew[0])

        if len(active) > 1:
            list(self._executor.map(
                lambda args: _step_single_env(*args),
                [(env_idx, local_i) for local_i, env_idx in enumerate(active)],
            ))
        else:
            _step_single_env(active[0], 0)

        # Update self._obs in-place for only the active envs — no full _stack_obs rebuild.
        # Inactive envs retain their previous raw obs without touching the numpy arrays.
        for local_i, env_idx in enumerate(active):
            ob = step_results[local_i]
            for key in self._obs:
                self._obs[key][env_idx] = ob[key][0]

        # Transform only the active envs, not the entire pool.
        # auto_reset (below) will call _transform_and_update_wrapped_obs for done envs only,
        # so those indices get exactly one transform total per step, not two.
        self._transform_and_update_wrapped_obs(list(active))

        done_idx, dones_bool = _get_dones(self._obs)
        # _get_dones scans the full backing batch (_total_envs). When operating on
        # a subset via get_subset(), keep only active envs so done_idx remains in
        # the same (underlying/global) index space as `active`.
        active_set = set(active)
        if len(done_idx) > 0:
            done_idx = np.array([i for i in done_idx if i in active_set], dtype=np.int32)

        # Per-env step count and step limit
        for env_idx in active:
            self._steps_since_reset[env_idx] += 1
        limit_idx = np.array(
            [
                i
                for i in range(self._total_envs)
                if self._steps_since_reset[i] >= self.max_episode_steps
            ],
            dtype=np.int32,
        )
        if len(limit_idx) > 0:
            done_idx = np.union1d(done_idx, limit_idx)
            dones_bool[limit_idx] = True

        # Action-loop detection: same K-length action sequence seen twice in episode
        loop_idx_list: List[int] = []
        if self.action_loop_enabled:
            for local_i, env_idx in enumerate(active):
                self._action_history[env_idx].append(int(actions[local_i]))
                if len(self._action_history[env_idx]) >= self.action_loop_window:
                    key = tuple(self._action_history[env_idx])
                    if key in self._action_hashes_seen[env_idx]:
                        loop_idx_list.append(env_idx)
                    else:
                        self._action_hashes_seen[env_idx].add(key)
            if loop_idx_list:
                loop_idx = np.array(loop_idx_list, dtype=np.int32)
                done_idx = np.union1d(done_idx, loop_idx)
                dones_bool[loop_idx] = True

        if termination_idx is None:
            termination_idx = [i for i, t in enumerate(self._terminated) if t]

        if self._game_mode == GameMode.PLAY_VS_OPPONENT:
            rewards = list(engine_rewards_full)
            raw_rewards = list(engine_rewards_full)
        else:
            rewards, raw_rewards = _calc_rewards(
                self.deck,
                self._wrapped_obs.decoded_cards,
                done_idx,
                self._claimed_reward_rules,
                termination_idx=termination_idx,
            )
        for i in limit_idx:
            rewards[i] = self.step_limit_penalty
        # Loop detection still sets done=True above; we no longer override reward (no -100 penalty)
        self._prev_rewards = raw_rewards

        # Reset done envs and slot new obs (skip when auto_reset=False so caller sees terminal state)
        if self._auto_reset:
            self.reset(env_indices=done_idx)
        # Map full-batch results back to the active subset for the caller.
        # Compute get_max_action_idx() once here (after potential auto-reset) rather than
        # calling it twice (the old code also called it at the top of step() — that result
        # is now stale after reset, so we only call it once at the end).
        subset_rewards = [rewards[i] for i in active]
        subset_raw_rewards = [raw_rewards[i] for i in active]
        subset_dones_bool = dones_bool[active]
        subset_done_idx = np.array(
            [local_i for local_i, env_idx in enumerate(active) if dones_bool[env_idx]],
            dtype=np.int32,
        )
        return (
            self.obs,
            subset_rewards,
            subset_dones_bool,
            subset_done_idx,
            subset_raw_rewards,
            self.get_max_action_idx(),
        )
