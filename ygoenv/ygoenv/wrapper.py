"""High-level batched YGO environment with explicit game modes."""

from __future__ import annotations

import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple, Union

import numpy as np

import ygoenv as ygoenv_lowlevel
from ygoenv.decoding import DoubleSliceableRecord, decode_all_batch
from ygoenv.init import init_ygopro
from ygoenv.modes import GameMode, OpponentMode, make_env_kwargs, resolve_mode_config
from ygoenv.paths import cards_db, deck_path

DEFAULT_MAX_CARDS = 80

SEED_MODE_FULL_DET = "full_det"
SEED_MODE_SEMI_DYNAMIC = "semi_dynamic"
SEED_MODE_FULL_DYNAMIC = "full_dynamic"
SEED_MODES = (SEED_MODE_FULL_DET, SEED_MODE_SEMI_DYNAMIC, SEED_MODE_FULL_DYNAMIC)


def _get_dones(obs: dict) -> Tuple[np.ndarray, np.ndarray]:
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
    if not obs_list:
        raise ValueError("obs_list must be non-empty")
    out = {}
    for key in obs_list[0].keys():
        parts = [o[key] for o in obs_list]
        out[key] = np.concatenate(parts, axis=0)
    return out


class YGOEnv:
    """Batched YGOPro environment with mode-based configuration.

    Owns N separate single-env ``ygoenv.make(num_envs=1)`` instances and
    aggregates observations.  Returns raw numpy obs dicts (not ML tensors).
    """

    def __init__(
        self,
        mode: GameMode,
        deck: Union[str, List[str]],
        num_envs: int,
        *,
        opponent_deck: str = "garnet",
        opponent_mode: OpponentMode = OpponentMode.BOT,
        greedy_reward: bool = True,
        ai_player: int = 0,
        seed_mode: str = SEED_MODE_FULL_DYNAMIC,
        base_seed: int = 0,
        db_path: Optional[str] = None,
        max_cards: int = DEFAULT_MAX_CARDS,
        verbose: bool = False,
        lang: str = "en",
    ) -> None:
        if seed_mode not in SEED_MODES:
            raise ValueError(f"seed_mode must be one of {SEED_MODES}, got {seed_mode!r}")

        self.mode = GameMode(mode) if isinstance(mode, str) else mode
        self.deck = deck
        self._total_envs = num_envs
        self.num_envs = num_envs
        self.opponent_deck = opponent_deck
        self._seed_mode = seed_mode
        self._base_seed = int(base_seed)
        self.max_cards = max_cards
        self.verbose = verbose
        self._mode_config = resolve_mode_config(
            self.mode,
            opponent_mode=opponent_mode,
            greedy_reward=greedy_reward,
            ai_player=ai_player,
        )

        db = db_path or str(cards_db(lang))
        if isinstance(deck, str):
            deck_names_in = [deck] * num_envs
        else:
            assert len(deck) == num_envs
            deck_names_in = list(deck)

        self._deck_names_per_env = deck_names_in
        unique_decks = list(dict.fromkeys(deck_names_in))
        skip_opponent = self.mode == GameMode.BOARD_SETUP
        player_name, registered = init_ygopro(
            [deck_path(d) for d in unique_decks],
            opponent_deck=None if skip_opponent else opponent_deck,
            db_path=db,
            return_deck_names=True,
        )
        if skip_opponent:
            opp_name = "_dummy"
        else:
            opp_stem = Path(opponent_deck).stem
            if opp_stem in registered:
                opp_name = opp_stem
            else:
                others = [n for n in registered if n != player_name]
                opp_name = others[0] if others else player_name

        self._envs: List[Any] = [None] * num_envs
        obs_list: List[Optional[dict]] = [None] * num_envs

        def _init_single(i: int) -> None:
            player_deck_name = Path(deck_names_in[i]).stem
            if self.mode == GameMode.PLAY_VS_OPPONENT:
                deck1, deck2 = player_deck_name, opp_name
            else:
                deck1, deck2 = player_deck_name, opp_name

            seed_kw: dict = {}
            if self._seed_mode == SEED_MODE_FULL_DET:
                seed_kw["seed"] = self._base_seed
            elif self._seed_mode == SEED_MODE_SEMI_DYNAMIC:
                seed_kw["seed"] = self._base_seed + i

            kwargs = make_env_kwargs(
                self.mode,
                deck1=deck1,
                deck2=deck2,
                max_cards=max_cards,
                opponent_mode=opponent_mode,
                greedy_reward=greedy_reward,
                ai_player=ai_player,
                verbose=verbose,
                seed=seed_kw.get("seed"),
            )
            env = ygoenv_lowlevel.make(**kwargs)
            reset_seed = self._get_reset_seed(i)
            if reset_seed is not None:
                obs, _ = env.reset(seed=reset_seed)
            else:
                obs, _ = env.reset()
            self._envs[i] = env
            obs_list[i] = obs

        workers = min(num_envs, os.cpu_count() or 1)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_init_single, range(num_envs)))
        else:
            for i in range(num_envs):
                _init_single(i)

        self._obs = _stack_obs(obs_list)
        self._executor = ThreadPoolExecutor(max_workers=max(num_envs, 1))
        self._active_indices: Optional[List[int]] = None
        self._decoded: Optional[DoubleSliceableRecord] = None
        self._refresh_decoded()

    @property
    def use_deck_rewards(self) -> bool:
        return self._mode_config.use_deck_rewards

    @property
    def decoded_cards(self) -> DoubleSliceableRecord:
        if self._decoded is None:
            self._refresh_decoded()
        assert self._decoded is not None
        return self._decoded

    def _refresh_decoded(self) -> None:
        batched = decode_all_batch(self._obs, True)
        self._decoded = DoubleSliceableRecord([d[0] for d in batched])

    @property
    def obs(self) -> dict:
        active = self._active_env_indices
        if len(active) == self._total_envs and active == list(range(self._total_envs)):
            return self._obs
        return {key: self._obs[key][active] for key in self._obs}

    @property
    def _active_env_indices(self) -> List[int]:
        if self._active_indices is None:
            return list(range(self._total_envs))
        return self._active_indices

    def get_subset(self, decks: List[str]) -> "YGOEnv":
        per_env = (
            [self.deck] * self._total_envs
            if isinstance(self.deck, str)
            else self.deck
        )
        selected = [i for i, d in enumerate(per_env) if d in decks]
        if not selected:
            raise ValueError(f"get_subset({decks!r}) matched no envs")
        self._active_indices = selected
        self.num_envs = len(selected)
        return self

    def _get_reset_seed(self, env_index: int) -> Optional[int]:
        if self._seed_mode == SEED_MODE_FULL_DET:
            return self._base_seed
        if self._seed_mode == SEED_MODE_SEMI_DYNAMIC:
            return self._base_seed + env_index
        return int(np.random.randint(0, 1_000_000))

    def set_seed_mode(self, seed_mode: str) -> None:
        if seed_mode not in SEED_MODES:
            raise ValueError(f"seed_mode must be one of {SEED_MODES}, got {seed_mode!r}")
        self._seed_mode = seed_mode

    def reset(
        self,
        env_indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> dict:
        if env_indices is None:
            env_indices = list(self._active_env_indices)
        elif self._active_indices is not None:
            active_set = set(self._active_env_indices)
            if not (set(env_indices) <= active_set):
                env_indices = [self._active_env_indices[i] for i in env_indices]

        lock = threading.Lock()

        def _reset_one(i: int) -> None:
            idx = int(i)
            reset_seed = (seed + idx) if seed is not None else self._get_reset_seed(idx)
            if reset_seed is not None:
                done_obs, _ = self._envs[idx].reset(seed=reset_seed)
            else:
                done_obs, _ = self._envs[idx].reset()
            with lock:
                for key in self._obs:
                    self._obs[key][idx] = done_obs[key][0]

        if len(env_indices) > 1:
            list(self._executor.map(_reset_one, env_indices))
        elif len(env_indices) == 1:
            _reset_one(env_indices[0])

        self._refresh_decoded()
        return self.obs

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[dict, List[float], np.ndarray, np.ndarray, List[float]]:
        """Step active envs.

        Returns
        -------
        obs, engine_rewards, dones_bool, done_idx, raw_engine_rewards
        """
        if isinstance(actions, list):
            actions = np.array(actions, dtype=np.int32)

        active = self._active_env_indices
        if actions.shape[0] != len(active):
            raise ValueError(
                f"actions shape {actions.shape[0]} != active envs {len(active)}"
            )

        engine_rewards_full = [0.0] * self._total_envs
        step_results: List[Optional[dict]] = [None] * len(active)

        def _step_one(env_idx: int, local_i: int) -> None:
            act = np.array([int(actions[local_i])], dtype=np.int32)
            ob, rew, term, trunc, _ = self._envs[env_idx].step(act)
            step_results[local_i] = ob
            engine_rewards_full[env_idx] = float(rew[0])

        if len(active) > 1:
            list(self._executor.map(
                lambda args: _step_one(*args),
                [(env_idx, li) for li, env_idx in enumerate(active)],
            ))
        else:
            _step_one(active[0], 0)

        for local_i, env_idx in enumerate(active):
            ob = step_results[local_i]
            assert ob is not None
            for key in self._obs:
                self._obs[key][env_idx] = ob[key][0]

        self._refresh_decoded()

        done_idx, dones_bool = _get_dones(self._obs)
        active_set = set(active)
        if len(done_idx) > 0:
            done_idx = np.array([i for i in done_idx if i in active_set], dtype=np.int32)

        subset_rewards = [engine_rewards_full[i] for i in active]
        return self.obs, subset_rewards, dones_bool[active], done_idx, subset_rewards

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        for env in self._envs:
            if env is not None and hasattr(env, "close"):
                env.close()

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
