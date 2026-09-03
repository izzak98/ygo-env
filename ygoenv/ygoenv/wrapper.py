"""High-level batched YGO environment with explicit game modes."""

from __future__ import annotations

import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

import ygoenv as ygoenv_lowlevel
from ygoenv.decoding import DoubleSliceableRecord, decode_all_batch
from ygoenv.init import init_ygopro
from ygoenv.modes import (GameMode, OpponentMode, make_env_kwargs, resolve_mode_config,
                          _native_has_ml_obs, validate_opening_hand,
                          _validate_reward_mode, _validate_episode_done_mode)
from ygoenv.paths import cards_db, deck_path

DEFAULT_MAX_CARDS = 80
OBS_FORMAT_RAW = "raw"
OBS_FORMAT_VECTORIZED = "vectorized"
OBS_FORMATS = (OBS_FORMAT_RAW, OBS_FORMAT_VECTORIZED)


def _filter_public_obs(obs: dict, obs_format: str) -> dict:
    if obs_format == OBS_FORMAT_VECTORIZED:
        return {k: v for k, v in obs.items() if k.startswith("ml_")}
    return {k: v for k, v in obs.items() if not k.startswith("ml_")}


def try_set_emb_index_map() -> None:
    """Push code_list → embedding-row indices into the native encoder (no-op if unavailable)."""
    try:
        from ygoenv.ygopro import set_emb_index_map
    except ImportError:
        return
    from ygoenv.decoding import get_code_list
    from ygoenv.env_wrapping.interface import _embedding_cache, load_embeddings
    from ygoenv.paths import embeddings_path

    if not embeddings_path().is_file():
        return
    embeddings = load_embeddings()
    _embedding_cache.init_fast(embeddings, get_code_list())
    set_emb_index_map(_embedding_cache._code_to_emb_idx.tolist())


SEED_MODE_FULL_DET = "full_det"
SEED_MODE_SEMI_DYNAMIC = "semi_dynamic"
SEED_MODE_FULL_DYNAMIC = "full_dynamic"
SEED_MODES = (SEED_MODE_FULL_DET, SEED_MODE_SEMI_DYNAMIC, SEED_MODE_FULL_DYNAMIC)


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

    ``opening_hand`` pins those cards into player 0's opening 5. Extra copies
    of the pinned codes are kept out of the remaining opening-hand slots.
    Pass a code twice to start with two copies. Entries may be int codes,
    numeric strings, or card names (looked up in cards.cdb).
    """

    def __init__(
        self,
        mode: GameMode,
        deck: Union[str, List[str]],
        num_envs: int,
        *,
        opponent_deck: str = "garnet",
        opponent_mode: OpponentMode = OpponentMode.BOT,
        ai_player: int = 0,
        seed_mode: str = SEED_MODE_FULL_DYNAMIC,
        base_seed: int = 0,
        db_path: Optional[str] = None,
        max_cards: int = DEFAULT_MAX_CARDS,
        verbose: bool = False,
        lang: str = "en",
        obs_format: str = OBS_FORMAT_RAW,
        opening_hand: Optional[Sequence[int | str]] = None,
        reward_mode: str = "duel",
        episode_done_mode: str = "duel",
    ) -> None:
        if seed_mode not in SEED_MODES:
            raise ValueError(f"seed_mode must be one of {SEED_MODES}, got {seed_mode!r}")
        if obs_format not in OBS_FORMATS:
            raise ValueError(f"obs_format must be one of {OBS_FORMATS}, got {obs_format!r}")
        _validate_reward_mode(reward_mode)
        _validate_episode_done_mode(episode_done_mode)

        self.mode = GameMode(mode) if isinstance(mode, str) else mode
        if obs_format == OBS_FORMAT_VECTORIZED and self.mode == GameMode.PLAY_VS_OPPONENT:
            raise ValueError("obs_format='vectorized' is not supported for play_vs_opponent yet")
        if obs_format == OBS_FORMAT_VECTORIZED and not _native_has_ml_obs():
            raise ValueError(
                "obs_format='vectorized' requires a ygopro_ygoenv.so build with compact ML obs keys"
            )

        self.obs_format = obs_format
        self.deck = deck
        self.opening_hand = list(opening_hand) if opening_hand else None
        self.reward_mode = reward_mode
        self.episode_done_mode = episode_done_mode
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
        if self.obs_format == OBS_FORMAT_VECTORIZED:
            try_set_emb_index_map()
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
            deck1 = player_deck_name
            deck2 = opp_name

            if self.opening_hand:
                validate_opening_hand(
                    self.opening_hand,
                    deck_ydk=str(deck_path(player_deck_name)),
                    db_path=db,
                )

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
                ai_player=ai_player,
                verbose=verbose,
                seed=seed_kw.get("seed"),
                obs_format=self.obs_format,
                opening_hand=self.opening_hand,
                db_path=db,
                reward_mode=self.reward_mode,
                episode_done_mode=self.episode_done_mode,
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
            full = self._obs
        else:
            full = {key: self._obs[key][active] for key in self._obs}
        return _filter_public_obs(full, self.obs_format)

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
        """Step active envs. Returns (obs, rewards, terminated, truncated, info)."""
        if isinstance(actions, list):
            actions = np.array(actions, dtype=np.int32)

        active = self._active_env_indices
        if actions.shape[0] != len(active):
            raise ValueError(
                f"actions shape {actions.shape[0]} != active envs {len(active)}"
            )

        engine_rewards_full = [0.0] * self._total_envs
        terminated_full = np.zeros(self._total_envs, dtype=bool)
        step_results: List[Optional[dict]] = [None] * len(active)

        def _step_one(env_idx: int, local_i: int) -> None:
            act = np.array([int(actions[local_i])], dtype=np.int32)
            ob, rew, term, trunc, _ = self._envs[env_idx].step(act)
            step_results[local_i] = ob
            engine_rewards_full[env_idx] = float(rew[0])
            terminated_full[env_idx] = bool(term[0]) or bool(trunc[0])

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

        subset_rewards = [engine_rewards_full[i] for i in active]
        subset_terminated = terminated_full[active]
        subset_truncated = np.zeros(len(active), dtype=bool)
        return self.obs, subset_rewards, subset_terminated, subset_truncated, {}

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
