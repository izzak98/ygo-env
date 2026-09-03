import torch
import numpy as np
import threading
from typing import Optional, Union
from ygoenv.decoding import (CardRecord,
                                   MSG_ID_MAP, ACT_CODE_MAP,
                                   CARD_OBS_BYTES,
                                   migrate_legacy_card_scale_rows,
                                   normalize_card_obs_width,
                                   _batch_decode_card_indices, _batch_decode_cards_raw,
                                   _batch_decode_actions_indices, _batch_decode_history_indices,
                                   get_code_list)
from ygoenv.constants import (DEVICE, EMBEDDING_SIZE, STATIC_CARD_DIM, STATIC_INTRINSIC_DIM,
                             STATIC_BOARD_DIM,
                             DYNAMIC_CARD_DIM, HISTORY_INFO_DIM, COMMAND_SIZE, ACT_SIZE,
                             ACTION_SIZE, CMND_INDICES, ACT_INDICES,
                             PROMPTS, LONG_PROMPTS_TO_SHORT_PROMPTS,
                             LINK_ARROW_BITS, CARD_SIZE)
from ygoenv.env_wrapping.lazy_intrinsic_static import get_lazy_intrinsic_table
from ygoenv.paths import embeddings_path as resolve_embeddings_path

import json
from pathlib import Path
from dataclasses import dataclass


# Feature indices
TYPE_INDICES = {
    "Monster": 1024,
    "Spell": 1025,
    "Trap": 1026,
    "Normal": 1027,
    "Effect": 1028,
    "Fusion": 1029,
    "Ritual": 1030,
    "Trap Monster": 1031,
    "Spirit": 1032,
    "Union": 1033,
    "Dual": 1034,
    "Tuner": 1035,
    "Synchro": 1036,
    "Token": 1037,
    "Quick-play": 1038,
    "Continuous": 1039,
    "Equip": 1040,
    "Field": 1041,
    "Counter": 1042,
    "Flip": 1043,
    "Toon": 1044,
    "XYZ": 1045,
    "Pendulum": 1046,
    "Link": 1047,
}

LOCATION_INDICES = {
    "Deck": 1051,
    "Hand": 1052,
    "Main Monster Zone": 1053,
    "Spell & Trap Zone": 1054,
    "Graveyard": 1055,
    "Banished": 1056,
    "Extra Deck": 1057,
}

POSITION_INDICES = {
    "face-up attack": 1066,
    "face-down attack": 1067,
    "attack": 1068,
    "face-up defense": 1069,
    "face-up": 1070,
    "face-down defense": 1071,
    "face-down": 1072,
    "defense": 1073,
}

ATTRIBUTE_INDICES = {
    "Earth": 1074,
    "Water": 1075,
    "Fire": 1076,
    "Wind": 1077,
    "Light": 1078,
    "Dark": 1079,
    "Divine": 1080,
}

RACE_INDICES = {
    "Warrior": 1081,
    "Spellcaster": 1082,
    "Fairy": 1083,
    "Fiend": 1084,
    "Zombie": 1085,
    "Machine": 1086,
    "Aqua": 1087,
    "Pyro": 1088,
    "Rock": 1089,
    "Windbeast": 1090,
    "Plant": 1091,
    "Insect": 1092,
    "Thunder": 1093,
    "Dragon": 1094,
    "Beast": 1095,
    "Beast Warrior": 1096,
    "Dinosaur": 1097,
    "Fish": 1098,
    "Sea Serpent": 1099,
    "Reptile": 1100,
    "Psycho": 1101,
    "Divine": 1102,
    "Creator God": 1103,
    "Wyrm": 1104,
    "Cyberse": 1105,
    "Illusion": 1106,
}

# ---------------------------------------------------------------------------
# O(1) index lookup tables — replaces list(DICT.keys()).index(key) O(n) calls
# These are built once at import time and used everywhere.
# ---------------------------------------------------------------------------
_TYPE_KEY_TO_IDX = {k: i for i, k in enumerate(TYPE_INDICES)}
_ATTR_KEY_TO_IDX = {k: i for i, k in enumerate(ATTRIBUTE_INDICES)}
_RACE_KEY_TO_IDX = {k: i for i, k in enumerate(RACE_INDICES)}
_POS_KEY_TO_IDX = {k: i for i, k in enumerate(POSITION_INDICES)}
_LOC_KEY_TO_IDX = {k: i for i, k in enumerate(LOCATION_INDICES)}

# ---------------------------------------------------------------------------
# Fast byte→index lookup tables for encode_all_batch_fast().
# Each maps a raw uint8 byte value (0-255) → feature column index (-1 = skip).
# Location/attribute/race/position bytes are 1-based indices into their string
# lists, so table[b] = b - 1 for valid range.
# ---------------------------------------------------------------------------
_LOC_BYTE_TO_IDX = np.full(256, -1, dtype=np.int16)
for _b in range(1, len(LOCATION_INDICES) + 1):
    _LOC_BYTE_TO_IDX[_b] = _b - 1

_ATTR_BYTE_TO_IDX = np.full(256, -1, dtype=np.int16)
for _b in range(1, len(ATTRIBUTE_INDICES) + 1):
    _ATTR_BYTE_TO_IDX[_b] = _b - 1

_RACE_BYTE_TO_IDX = np.full(256, -1, dtype=np.int16)
for _b in range(1, len(RACE_INDICES) + 1):
    _RACE_BYTE_TO_IDX[_b] = _b - 1

_POS_BYTE_TO_IDX = np.full(256, -1, dtype=np.int16)
for _b in range(1, len(POSITION_INDICES) + 1):
    _POS_BYTE_TO_IDX[_b] = _b - 1

# msg_id byte → CMND_INDICES value, act_code byte → ACT_INDICES value
_MSG_ID_TO_CMND_IDX = np.full(256, -1, dtype=np.int16)
for _mid, _mname in MSG_ID_MAP.items():
    _v = CMND_INDICES.get(_mname, -1)
    if 0 <= _mid <= 255:
        _MSG_ID_TO_CMND_IDX[_mid] = _v

_ACT_CODE_TO_ACT_IDX = np.full(256, -1, dtype=np.int16)
for _acode, _aname in ACT_CODE_MAP.items():
    if _aname is not None:
        _v = ACT_INDICES.get(_aname, -1)
        if 0 <= _acode <= 255:
            _ACT_CODE_TO_ACT_IDX[_acode] = _v

# msg_id byte → PROMPTS index (for prompt_one_hot)
_MSG_ID_TO_PROMPT_IDX = np.full(256, -1, dtype=np.int16)
for _mid, _mname in MSG_ID_MAP.items():
    _short = LONG_PROMPTS_TO_SHORT_PROMPTS.get(_mname)
    if _short is not None and _short in PROMPTS:
        _MSG_ID_TO_PROMPT_IDX[_mid] = PROMPTS.index(_short)

# Message types
SPEC_INDEX_MESSAGES = {"SELECT_IDLECMD", "SELECT_CHAIN",
                       "SELECT_EFFECTYN", "SELECT_UNSELECT_CARD", "SELECT_SUM", "SELECT_TRIBUTE"}
CARD_ID_MESSAGES = {"SELECT_CARD", "SELECT_YESNO", "SELECT_OPTION"}
SKIP_MESSAGES = {"SELECT_PLACE", "SELECT_POSITION"}

# Integer sets for action type dispatch in encode_all_batch_fast()
_SKIP_MSG_IDS = frozenset(
    mid for mid, name in MSG_ID_MAP.items() if name in SKIP_MESSAGES)
_SPEC_INDEX_MSG_IDS = frozenset(
    mid for mid, name in MSG_ID_MAP.items() if name in SPEC_INDEX_MESSAGES)
_CARD_ID_MSG_IDS = frozenset(
    mid for mid, name in MSG_ID_MAP.items() if name in CARD_ID_MESSAGES)
_CANCEL_ACT_CODE = 9   # ACT_CODE_MAP[9] == "Cancel"
_EMPTY_INT32 = np.array([], dtype=np.int32)


# ============================================================================
# PINNED MEMORY BUFFERS (for fast async H2D transfers)
# ============================================================================

class _PinnedBuffers:
    """Persistent pinned-CPU + pre-allocated GPU buffer pairs for encode_all_batch_fast.

    Uses double-buffering on the CPU side: two pinned-memory slots (A and B) are
    alternated on each call.  While the GPU DMA is reading from slot A, the CPU
    is zeroing and filling slot B.  This eliminates the ~270ms OS/hardware stall
    that occurs when the CPU writes to pinned memory immediately after it has been
    used as a DMA source (TLB / page-table side-effects of PCIe DMA).

    Using pinned CPU memory and preallocated GPU tensors with copy_() eliminates:
    1. New GPU tensor allocation every step (causes ~200ms PyTorch allocator stalls)
    2. Synchronous H2D transfer (non_blocking=True requires pinned source)
    3. Per-step pin_memory() calls (expensive OS mlock)

    The DMA runs on the CUDA default stream; CPU returns immediately.
    GPU tensors are synchronised automatically when the next CUDA kernel uses them.

    Buffers are keyed by (key, trailing_dims) and sized to the largest batch_size
    seen so far.  Smaller batches get views into the first batch_size rows.
    """

    def __init__(self):
        self._cpu: list[dict] = [{}, {}]   # two pinned CPU tensor slots
        self._gpu: dict = {}               # preallocated GPU tensors (one per key)
        self._is_cuda: bool = str(DEVICE).startswith("cuda")
        self._slot: int = 0                # current active CPU slot (0 or 1)

    @staticmethod
    def _is_inference_tensor(t: torch.Tensor) -> bool:
        # `torch.inference_mode()` can create "inference tensors" which cannot be
        # mutated in-place outside inference mode. If these get cached and later
        # reused in training mode, .zero_() / .copy_() will crash.
        is_inf = getattr(t, "is_inference", None)
        return bool(is_inf()) if callable(is_inf) else False

    def next_slot(self) -> None:
        """Advance to the other CPU slot (call once at the start of each encode)."""
        self._slot ^= 1

    def get_cpu(self, key: str, shape: tuple, dtype=torch.float32) -> torch.Tensor:
        """Return (or lazily grow) a pinned CPU tensor for the current slot."""
        trailing = shape[1:]
        bk = (key, trailing, dtype)
        slot = self._cpu[self._slot]
        if not self._is_cuda:
            if bk in slot and self._is_inference_tensor(slot[bk]):
                # Drop cached inference tensor; recreate as a normal tensor.
                slot.pop(bk, None)
            if bk not in slot or slot[bk].shape[0] < shape[0]:
                slot[bk] = torch.empty(shape, dtype=dtype)
            return slot[bk][:shape[0]]
        if bk in slot and self._is_inference_tensor(slot[bk]):
            # Drop cached inference tensor; recreate as a normal pinned tensor.
            slot.pop(bk, None)
        if bk not in slot or slot[bk].shape[0] < shape[0]:
            buf = torch.empty(shape, dtype=dtype).pin_memory()
            buf.zero_()  # pre-warm: first-touch all pages now to avoid OS page-fault stall later
            slot[bk] = buf
        return slot[bk][:shape[0]]

    def get_gpu(self, key: str, shape: tuple, dtype=torch.float32) -> torch.Tensor:
        """Return (or lazily grow) a preallocated GPU tensor at least as large as shape."""
        trailing = shape[1:]
        bk = (key, trailing, dtype)
        if bk in self._gpu and self._is_inference_tensor(self._gpu[bk]):
            # Drop cached inference tensor; recreate as a normal tensor.
            self._gpu.pop(bk, None)
        if bk not in self._gpu or self._gpu[bk].shape[0] < shape[0]:
            self._gpu[bk] = torch.empty(shape, dtype=dtype, device=DEVICE)
        return self._gpu[bk][:shape[0]]


_pinned_buffers = _PinnedBuffers()

# Persistent GPU tensors for partial-batch (use_preallocated_gpu=False) encode calls.
# These avoid per-call cudaMalloc which causes ~50ms device syncs per allocation.
# Keyed by (name, trailing_shape, dtype) → allocated once, reused every reset step.
_partial_gpu: dict = {}


def _get_partial_gpu(key: str, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return a persistent GPU tensor for partial-batch use (lazily allocated)."""
    trailing = shape[1:]
    bk = (key, trailing, dtype)
    if bk not in _partial_gpu or _partial_gpu[bk].shape[0] < shape[0]:
        _partial_gpu[bk] = torch.empty(shape, dtype=dtype, device=DEVICE)
    return _partial_gpu[bk][:shape[0]]


# ============================================================================
# EMBEDDING CACHE
# ============================================================================

class EmbeddingCache:
    """Cache for embedding lookups with pre-converted arrays."""

    def __init__(self):
        self._embeddings_dict = None
        self._embedding_array = None
        self._name_embedding_array = None
        self._card_id_to_idx = None
        self._lock = threading.Lock()
        self._code_to_emb_idx = None   # built by init_fast()

    def initialize(self, embeddings: dict):
        """Initialize the cache with embeddings dictionary (thread-safe)."""
        # Fast path: already initialized
        if self._embeddings_dict is embeddings and self._name_embedding_array is not None:
            return

        # Acquire lock for initialization
        with self._lock:
            # Double-check after acquiring lock
            if self._embeddings_dict is embeddings and self._name_embedding_array is not None:
                return

            self._embeddings_dict = embeddings

            # Build index mapping
            card_ids = list(embeddings.keys())
            self._card_id_to_idx = {card_id: idx for idx, card_id in enumerate(card_ids)}

            # Pre-convert all embeddings to numpy arrays
            embedding_list = [embeddings[card_id]["embedding"] for card_id in card_ids]
            self._embedding_array = np.array(embedding_list, dtype=np.float32)

            # Pre-convert name embeddings if available
            if "name_embedding" in embeddings[card_ids[0]]:
                name_embedding_list = [embeddings[card_id]["name_embedding"]
                                       for card_id in card_ids]
                self._name_embedding_array = np.array(name_embedding_list, dtype=np.float32)
            else:
                # Fallback: use description embedding if name_embedding not available
                self._name_embedding_array = self._embedding_array

    def get_embeddings_batch(self, card_ids: list[str]) -> np.ndarray:
        """Get description embeddings for multiple card IDs at once."""
        indices = [self._card_id_to_idx[cid] for cid in card_ids]
        return self._embedding_array[indices]

    def get_name_embeddings_batch(self, card_ids: list[str]) -> np.ndarray:
        """Get name embeddings for multiple card IDs at once."""
        indices = [self._card_id_to_idx[cid] for cid in card_ids]
        return self._name_embedding_array[indices]

    def get_embedding(self, card_id: str) -> np.ndarray:
        """Get description embedding for a single card ID."""
        idx = self._card_id_to_idx[card_id]
        return self._embedding_array[idx]

    def get_name_embedding(self, card_id: str) -> np.ndarray:
        """Get name embedding for a single card ID."""
        idx = self._card_id_to_idx[card_id]
        return self._name_embedding_array[idx]

    def init_fast(self, embeddings: dict, code_list: list) -> None:
        """Extend initialize() with a code_list-index → embedding-array-index mapping.

        Enables encode_all_batch_fast() to do purely numpy fancy-index lookups
        instead of per-card dict calls.  Built once; subsequent calls are no-ops.
        """
        self.initialize(embeddings)
        if self._code_to_emb_idx is not None:
            return
        arr = np.full(len(code_list), -1, dtype=np.int32)
        for ci, cid in enumerate(code_list):
            arr[ci] = self._card_id_to_idx.get(cid, -1)
        self._code_to_emb_idx = arr

    def get_gpu_tables(self) -> "tuple[torch.Tensor, torch.Tensor]":
        """Return (name_emb_gpu, desc_emb_gpu) — lazily uploaded to DEVICE.

        These are persistent GPU tensors for the full embedding vocabulary.  They are
        used for GPU-side index lookups, eliminating the 200 MB CPU→GPU DMA that
        caused the pinned-memory stall on every step.
        """
        if not hasattr(self, "_name_embedding_gpu"):
            self._name_embedding_gpu: Optional[torch.Tensor] = None
            self._embedding_gpu: Optional[torch.Tensor] = None
        if self._name_embedding_gpu is None and str(DEVICE).startswith("cuda"):
            self._name_embedding_gpu = torch.from_numpy(
                self._name_embedding_array).to(DEVICE)
            self._embedding_gpu = torch.from_numpy(
                self._embedding_array).to(DEVICE)
        return self._name_embedding_gpu, self._embedding_gpu


_embedding_cache = EmbeddingCache()


# ============================================================================
# VECTORIZED CARD ENCODING (used by intrinsic_static_table.py)
# ============================================================================

def _encode_cards_vectorized(cards: list[CardRecord], embeddings: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized encoding of multiple cards at once.

    Returns:
        tuple: (name_embeddings, desc_embeddings, static_features, dynamic_features)
        - name_embeddings: (n_cards, EMBEDDING_SIZE) name embeddings
        - desc_embeddings: (n_cards, EMBEDDING_SIZE) description embeddings
        - static_features: (n_cards, STATIC_CARD_DIM) static card properties
        - dynamic_features: (n_cards, DYNAMIC_CARD_DIM) location + actionability
    """
    n_cards = len(cards)
    if n_cards == 0:
        return (
            np.zeros((0, EMBEDDING_SIZE), dtype=np.float32),
            np.zeros((0, EMBEDDING_SIZE), dtype=np.float32),
            np.zeros((0, STATIC_CARD_DIM), dtype=np.float32),
            np.zeros((0, DYNAMIC_CARD_DIM), dtype=np.float32),
        )

    _embedding_cache.initialize(embeddings)

    card_ids = [card.card_id for card in cards]
    name_embeddings = _embedding_cache.get_name_embeddings_batch(card_ids)
    desc_embeddings = _embedding_cache.get_embeddings_batch(card_ids)

    static_features = np.zeros((n_cards, STATIC_CARD_DIM), dtype=np.float32)
    dynamic_features = np.zeros((n_cards, DYNAMIC_CARD_DIM), dtype=np.float32)

    types_list = [card.types for card in cards]
    locations = [card.location for card in cards]
    positions = [card.position for card in cards]
    sequences = [card.seq for card in cards]
    overlays = [card.overlay for card in cards]

    is_monster = np.array(['Monster' in types for types in types_list])
    levels = np.array([card.level if is_monster[i] else 0 for i,
                      card in enumerate(cards)], dtype=np.float32)
    atks = np.array([card.atk_norm if is_monster[i] else 0 for i,
                    card in enumerate(cards)], dtype=np.float32)
    defs = np.array([card.def_norm if is_monster[i] else 0 for i,
                    card in enumerate(cards)], dtype=np.float32)
    attributes = [card.attribute if is_monster[i] else None for i, card in enumerate(cards)]
    races = [card.race if is_monster[i] else None for i, card in enumerate(cards)]

    # Types: O(1) lookup
    for i, types in enumerate(types_list):
        for t in types:
            type_idx = _TYPE_KEY_TO_IDX.get(t)
            if type_idx is not None and type_idx < 24:
                static_features[i, type_idx] = 1.0

    link_type_idx = _TYPE_KEY_TO_IDX.get("Link")
    is_link = np.array(['Link' in types for types in types_list])

    # Level, ATK, DEF
    stat_offset = 24
    monster_mask = is_monster
    static_features[monster_mask, stat_offset] = levels[monster_mask] / 13
    static_features[monster_mask, stat_offset + 1] = atks[monster_mask]
    # Link monsters transport arrow bitmask in defense; encode def as 0.0.
    def_vals = defs.copy()
    def_vals[is_link] = 0.0
    static_features[monster_mask, stat_offset + 2] = def_vals[monster_mask]

    # Attributes: O(1) lookup
    attr_offset = 27
    for i in np.where(monster_mask)[0]:
        if attributes[i]:
            attr_idx = _ATTR_KEY_TO_IDX.get(attributes[i])
            if attr_idx is not None and attr_idx < 7:
                static_features[i, attr_offset + attr_idx] = 1.0

    # Races: O(1) lookup
    race_offset = 34
    for i in np.where(monster_mask)[0]:
        if races[i]:
            race_idx = _RACE_KEY_TO_IDX.get(races[i])
            if race_idx is not None and race_idx < 26:
                static_features[i, race_offset + race_idx] = 1.0

    # Link arrows: 8 one-hot columns (intrinsic).
    # Column order matches LINK_ARROW_BITS.
    link_arrow_offset = 60
    if link_type_idx is not None:
        # CardRecord.def_raw is expected to carry the link marker bitmask for Link monsters.
        for i in np.where(is_link)[0]:
            bitmask = int(getattr(cards[i], "def_raw", 0) or 0)
            for j, bit in enumerate(LINK_ARROW_BITS):
                if bitmask & bit:
                    static_features[i, link_arrow_offset + j] = 1.0

    # Position: O(1) lookup
    pos_offset = 68
    for i, pos in enumerate(positions):
        if pos:
            pos_idx = _POS_KEY_TO_IDX.get(pos)
            if pos_idx is not None and pos_idx < 8:
                static_features[i, pos_offset + pos_idx] = 1.0

    # Sequence: one-hot (8 possible, 0-7)
    seq_offset = 76
    for i, loc in enumerate(locations):
        if loc in ("Main Monster Zone", "Spell & Trap Zone"):
            seq = sequences[i]
            if seq is not None and 0 <= seq <= 7:
                static_features[i, seq_offset + seq] = 1.0

    # Overlay: binary flag
    overlay_offset = 84
    for i, overlay in enumerate(overlays):
        if overlay:
            static_features[i, overlay_offset] = 1.0

    # Dynamic features: Location (O(1) lookup)
    for i, loc in enumerate(locations):
        if loc:
            loc_idx = _LOC_KEY_TO_IDX.get(loc)
            if loc_idx is not None and loc_idx < 7:
                dynamic_features[i, loc_idx] = 1.0

    return name_embeddings, desc_embeddings, static_features, dynamic_features


# ============================================================================
# MAIN ENCODING FUNCTION
# ============================================================================

@dataclass
class EncodedObsCompact:
    """Compact encoded observation: embedding indices + board-state features only.

    Intrinsic static columns (0-(STATIC_INTRINSIC_DIM-1)) are stored in a process-wide lazy table
    and reconstructed in expand_obs_for_encoder / reconstruct_full_card_static.
    """

    card_emb_idx: torch.Tensor      # (B, max_cards) int64, -1 = pad
    hist_emb_idx: torch.Tensor      # (B, max_histories) int64
    # (B, max_cards, STATIC_BOARD_DIM) cols STATIC_INTRINSIC_DIM..(STATIC_CARD_DIM-1)
    card_static_board: torch.Tensor
    card_dynamic: torch.Tensor
    history_info: torch.Tensor
    shuffle_indices: torch.Tensor


def encode_all_batch_fast(
    obs_batch: dict,
    embeddings: dict,
    max_cards: int = CARD_SIZE,
    max_histories: int = 300,
    use_preallocated_gpu: bool = True,
) -> tuple[EncodedObsCompact, torch.Tensor]:
    """Fast vectorized encoder — bypasses Python object creation entirely.

    Uses prebuilt byte→index lookup tables instead of dict lookups, vectorised
    numpy fancy-indexing for all card/history features, and pure numpy embedding
    lookups (no string → dict → index indirection).

    Returns (EncodedObsCompact, prompt_one_hot_tensor).

    NOTE: First call may be slow due to Numba JIT compilation.
    """
    cards_bytes = obs_batch["cards_"]       # [B, n_raw, CARD_OBS_BYTES]
    actions_bytes = obs_batch["actions_"]     # [B, n_act, 12]
    history_bytes = obs_batch["h_actions_"]   # [B, n_hist, 14]

    if not isinstance(cards_bytes, np.ndarray):
        cards_bytes = np.array(cards_bytes)
    if not isinstance(actions_bytes, np.ndarray):
        actions_bytes = np.array(actions_bytes)
    if not isinstance(history_bytes, np.ndarray):
        history_bytes = np.array(history_bytes)

    cards_bytes = normalize_card_obs_width(cards_bytes)

    if cards_bytes.shape[-1] < CARD_OBS_BYTES:
        pad_w = CARD_OBS_BYTES - int(cards_bytes.shape[-1])
        cards_bytes = np.pad(
            cards_bytes,
            ((0, 0),) * (cards_bytes.ndim - 1) + ((0, pad_w),),
            mode="constant",
        )
    migrate_legacy_card_scale_rows(cards_bytes)

    batch_size = cards_bytes.shape[0]
    code_list = get_code_list()
    n_codes = len(code_list)

    card_indices = _batch_decode_card_indices(cards_bytes, n_codes)    # [B, n_raw]
    raw_fields = _batch_decode_cards_raw(cards_bytes)
    action_indices = _batch_decode_actions_indices(actions_bytes, n_codes)  # [B, n_act]
    history_indices = _batch_decode_history_indices(history_bytes, n_codes)  # [B, n_hist]

    (locations, seqs, owners, positions, overlays,
     attributes, races, levels, _counters, _negateds,
     atks, defs, types_flags, _pscales) = raw_fields

    _embedding_cache.init_fast(embeddings, code_list)

    card_emb_idx_np = np.full((batch_size, max_cards), -1, dtype=np.int64)
    hist_emb_idx_np = np.full((batch_size, max_histories), -1, dtype=np.int64)

    # Pre-allocate outputs using double-buffered pinned-CPU + pre-allocated GPU tensor pairs.
    # Double-buffering: after GPU copy we advance to the OTHER slot, so the next call zeros
    # a slot whose DMA completed at least one full step ago.  We only advance the slot for
    # full-batch (use_preallocated_gpu=True) calls — partial-batch calls do no GPU DMA, so
    # advancing for them would cause the next full-batch call to accidentally zero the slot
    # that was just DMA'd (the "double-flip" regression).
    n_prompts = len(PROMPTS)
    _pb = _pinned_buffers
    if use_preallocated_gpu:
        _card_static_buf = _pb.get_cpu(
            "card_static",     (batch_size, max_cards,     STATIC_CARD_DIM))
        _card_dynamic_buf = _pb.get_cpu(
            "card_dynamic",    (batch_size, max_cards,     DYNAMIC_CARD_DIM))
        _hist_info_buf = _pb.get_cpu(
            "hist_info",       (batch_size, max_histories, HISTORY_INFO_DIM))
        _shuf_buf = _pb.get_cpu("shuffle_indices", (batch_size, max_cards),    torch.int64)
        _prompt_buf = _pb.get_cpu("prompt_one_hot",  (batch_size, n_prompts))

        _card_static_board_gpu = _pb.get_gpu(
            "card_static_board", (batch_size, max_cards, STATIC_BOARD_DIM))
        _card_dynamic_gpu = _pb.get_gpu(
            "card_dynamic",    (batch_size, max_cards,     DYNAMIC_CARD_DIM))
        _hist_info_gpu = _pb.get_gpu(
            "hist_info",       (batch_size, max_histories, HISTORY_INFO_DIM))
        _shuf_gpu = _pb.get_gpu("shuffle_indices", (batch_size, max_cards),    torch.int64)
        _prompt_gpu = _pb.get_gpu("prompt_one_hot",  (batch_size, n_prompts))
    else:
        # Partial-batch: plain non-pinned CPU tensors for small features.
        _card_static_buf = torch.empty((batch_size, max_cards,     STATIC_CARD_DIM))
        _card_dynamic_buf = torch.empty((batch_size, max_cards,     DYNAMIC_CARD_DIM))
        _hist_info_buf = torch.empty((batch_size, max_histories, HISTORY_INFO_DIM))
        _shuf_buf = torch.empty((batch_size, max_cards),    dtype=torch.int64)
        _prompt_buf = torch.empty((batch_size, n_prompts))

    # Zero-fill small CPU buffers — only ~5 MB total, negligible time, no DMA stall.
    _card_static_buf.zero_()
    _card_dynamic_buf.zero_()
    _hist_info_buf.zero_()
    _shuf_buf.fill_(-1)
    _prompt_buf.zero_()

    # Numpy views into the small CPU buffers (zero-copy).
    card_static_np = _card_static_buf.numpy()
    card_dynamic_np = _card_dynamic_buf.numpy()
    history_info_np = _hist_info_buf.numpy()
    shuffle_indices_np = _shuf_buf.numpy()
    prompt_one_hot_np = _prompt_buf.numpy()

    # ================================================================
    # CARDS — build flat (env, me-pos, raw-pos) index triplets
    # ================================================================
    # me-card mask: non-zero slot + my card + valid code
    raw_nonzero = cards_bytes.any(axis=2)                       # [B, n_raw]
    me_mask = (owners == 0) & raw_nonzero & (card_indices >= 0)  # [B, n_raw]

    me_raw_per_env: list = []
    ei_parts, pi_parts, ri_parts = [], [], []
    for b in range(batch_size):
        me_raw = np.where(me_mask[b])[0][:max_cards].astype(np.int32)
        me_raw_per_env.append(me_raw)
        n = len(me_raw)
        if n:
            ei_parts.append(np.full(n, b, np.int32))
            pi_parts.append(np.arange(n, dtype=np.int32))
            ri_parts.append(me_raw)
            shuffle_indices_np[b, :n] = np.arange(n)

    if ei_parts:
        ei = np.concatenate(ei_parts)   # flat env index
        pi = np.concatenate(pi_parts)   # flat me-card position
        ri = np.concatenate(ri_parts)   # flat raw card position

        # --- Embeddings ---
        code_idx = card_indices[ei, ri]                         # [n_total]
        emb_idx = _embedding_cache._code_to_emb_idx[code_idx]  # [n_total]
        ve = emb_idx >= 0

        if ve.any():
            card_emb_idx_np[ei[ve], pi[ve]] = emb_idx[ve].astype(np.int64)

        # --- Card static: types (columns 0-23) ---
        card_static_np[ei, pi, :24] = types_flags[ei, ri, :24].astype(np.float32)

        # --- Monster-only features ---
        is_monster = types_flags[ei, ri, 0]                    # bool [n_total]
        if is_monster.any():
            m_ei, m_pi, m_ri = ei[is_monster], pi[is_monster], ri[is_monster]
            card_static_np[m_ei, m_pi, 24] = levels[m_ei, m_ri].astype(np.float32) / 13.0
            card_static_np[m_ei, m_pi, 25] = atks[m_ei, m_ri].astype(np.float32) / 65535.0
            # Link monsters: transport arrow bitmask in defense; encode DEF as 0.0.
            link_idx = _TYPE_KEY_TO_IDX.get("Link")
            if link_idx is not None:
                is_link_m = types_flags[m_ei, m_ri, link_idx]
            else:
                is_link_m = np.zeros(m_ei.shape[0], dtype=bool)
            def_norm = defs[m_ei, m_ri].astype(np.float32) / 65535.0
            def_norm[is_link_m] = 0.0
            card_static_np[m_ei, m_pi, 26] = def_norm

            attr_b = attributes[m_ei, m_ri]
            attr_idx = _ATTR_BYTE_TO_IDX[attr_b].astype(np.int32)
            va = (attr_idx >= 0) & (attr_idx < 7)
            if va.any():
                card_static_np[m_ei[va], m_pi[va], 27 + attr_idx[va]] = 1.0

            race_b = races[m_ei, m_ri]
            race_idx = _RACE_BYTE_TO_IDX[race_b].astype(np.int32)
            vr = (race_idx >= 0) & (race_idx < 26)
            if vr.any():
                card_static_np[m_ei[vr], m_pi[vr], 34 + race_idx[vr]] = 1.0

        # --- Link arrows (columns 60-67) ---
        link_idx2 = _TYPE_KEY_TO_IDX.get("Link")
        if link_idx2 is not None:
            is_link = types_flags[ei, ri, link_idx2]
            if is_link.any():
                l_ei, l_pi, l_ri = ei[is_link], pi[is_link], ri[is_link]
                # defs[...] holds the raw defense field; for Link monsters this is the arrow bitmask.
                mask_vals = defs[l_ei, l_ri].astype(np.int32)
                for j, bit in enumerate(LINK_ARROW_BITS):
                    on = (mask_vals & bit) != 0
                    if on.any():
                        card_static_np[l_ei[on], l_pi[on], 60 + j] = 1.0

        # --- Position (columns 68-75) ---
        pos_b = positions[ei, ri]
        pos_idx = _POS_BYTE_TO_IDX[pos_b].astype(np.int32)
        vp = (pos_idx >= 0) & (pos_idx < 8)
        if vp.any():
            card_static_np[ei[vp], pi[vp], 68 + pos_idx[vp]] = 1.0

        # --- Sequence in MMZ / STZ (columns 76-83) ---
        # Location bytes are 1-based: MMZ=3, STZ=4
        loc_b = locations[ei, ri]
        in_zone = (loc_b == 3) | (loc_b == 4)
        if in_zone.any():
            seq_v = seqs[ei[in_zone], ri[in_zone]].astype(np.int32)
            vs = (seq_v >= 1) & (seq_v <= 7)
            if vs.any():
                card_static_np[ei[in_zone][vs], pi[in_zone][vs], 76 + seq_v[vs]] = 1.0

        # --- Overlay flag (column 84) ---
        ov = overlays[ei, ri]
        if ov.any():
            card_static_np[ei[ov], pi[ov], 84] = 1.0

        # --- Card dynamic: location one-hot (columns 0-6) ---
        loc_idx = _LOC_BYTE_TO_IDX[loc_b].astype(np.int32)
        vl = (loc_idx >= 0) & (loc_idx < 7)
        if vl.any():
            card_dynamic_np[ei[vl], pi[vl], loc_idx[vl]] = 1.0

    # ================================================================
    # ACTIONS — per-env loop (no Python objects; numpy for targets)
    # ================================================================
    n_raw_actions = actions_bytes.shape[1]
    for b in range(batch_size):
        first_msg_id = int(actions_bytes[b, 0, 3])
        if first_msg_id == 0 or first_msg_id in _SKIP_MSG_IDS:
            continue
        me_raw = me_raw_per_env[b]
        n_me = len(me_raw)
        me_codes = card_indices[b, me_raw] if n_me > 0 else _EMPTY_INT32
        for a in range(n_raw_actions):
            msg_id = int(actions_bytes[b, a, 3])
            if msg_id == 0:
                break
            act_code = int(actions_bytes[b, a, 4])
            if act_code == _CANCEL_ACT_CODE:
                continue

            action_vec = np.zeros(ACTION_SIZE, np.float32)
            cmd_idx = int(_MSG_ID_TO_CMND_IDX[msg_id])
            if cmd_idx >= 0:
                action_vec[cmd_idx] = 1.0
            act_idx = int(_ACT_CODE_TO_ACT_IDX[act_code])
            if act_idx >= 0:
                action_vec[act_idx] = 1.0

            if msg_id in _SPEC_INDEX_MSG_IDS:
                spec = int(actions_bytes[b, a, 0]) - 1
                if 0 <= spec < n_me:
                    card_dynamic_np[b, spec, 7:7 + COMMAND_SIZE] = action_vec[:COMMAND_SIZE]
                    card_dynamic_np[b, spec, 7 + COMMAND_SIZE:7 +
                                    ACTION_SIZE] += action_vec[COMMAND_SIZE:]
            elif msg_id in _CARD_ID_MSG_IDS:
                act_card_code = int(action_indices[b, a])
                if act_card_code >= 0 and n_me > 0:
                    tgts = np.where(me_codes == act_card_code)[0]
                    if tgts.size:
                        card_dynamic_np[b, tgts, 7:7 + COMMAND_SIZE] = action_vec[:COMMAND_SIZE]
                        card_dynamic_np[b, tgts, 7 + COMMAND_SIZE:7 +
                                        ACTION_SIZE] += action_vec[COMMAND_SIZE:]

    # Cap action band (command/action one-hots) at 1.0.
    np.clip(card_dynamic_np[:, :, 7:7 + ACTION_SIZE], 0.0,
            1.0, out=card_dynamic_np[:, :, 7:7 + ACTION_SIZE])

    # ================================================================
    # PROMPT ONE-HOT — fully vectorised
    # ================================================================
    act_valid_mask = actions_bytes.any(axis=2)                        # [B, n_act]
    msg_ids_arr = actions_bytes[:, :, 3].astype(np.uint8)
    pidx_arr = _MSG_ID_TO_PROMPT_IDX[msg_ids_arr]              # [B, n_act] int16
    valid_prompt = act_valid_mask & (pidx_arr >= 0)
    if valid_prompt.any():
        vb_p, va_p = np.where(valid_prompt)
        np.add.at(prompt_one_hot_np, (vb_p, pidx_arr[vb_p, va_p].astype(np.int32)), 1.0)
        np.clip(prompt_one_hot_np, 0.0, 1.0, out=prompt_one_hot_np)

    # ================================================================
    # HISTORIES — fully vectorised
    # ================================================================
    hist_valid = history_bytes.any(axis=2)   # [B, n_hist]
    h_b, h_j = np.where(hist_valid)
    if h_b.size:
        in_range = h_j < max_histories
        h_b, h_j = h_b[in_range], h_j[in_range]

    if h_b.size:
        h_codes = history_indices[h_b, h_j]
        v_mask = h_codes >= 0
        if v_mask.any():
            v_hb, v_hj = h_b[v_mask], h_j[v_mask]
            h_emb_idx = _embedding_cache._code_to_emb_idx[h_codes[v_mask]]
            ve = h_emb_idx >= 0
            if ve.any():
                hist_emb_idx_np[v_hb[ve], v_hj[ve]] = h_emb_idx[ve].astype(np.int64)

        h_msg = history_bytes[h_b, h_j, 3].astype(np.uint8)
        h_act = history_bytes[h_b, h_j, 4].astype(np.uint8)
        cmnd = _MSG_ID_TO_CMND_IDX[h_msg].astype(np.int32)
        aact = _ACT_CODE_TO_ACT_IDX[h_act].astype(np.int32)

        vc = (cmnd >= 0) & (cmnd < COMMAND_SIZE)
        if vc.any():
            history_info_np[h_b[vc], h_j[vc], cmnd[vc]] = 1.0

        va_h = (aact >= 11) & (aact < 11 + ACT_SIZE)
        if va_h.any():
            history_info_np[h_b[va_h], h_j[va_h], COMMAND_SIZE + aact[va_h] - 11] = 1.0

    # ================================================================
    # Lazy intrinsic static: learn [:STATIC_INTRINSIC_DIM] rows from this encode.
    # ================================================================
    get_lazy_intrinsic_table().update_from_numpy(card_emb_idx_np, card_static_np)

    # ================================================================
    # Transfer small feature tensors to device.
    # Full-batch: async DMA from pinned CPU → preallocated GPU (only ~5 MB, no stall).
    # Partial-batch: synchronous copy of non-pinned CPU tensors (small batch, fast).
    # ================================================================
    if use_preallocated_gpu:
        _board_np = np.ascontiguousarray(
            card_static_np[:, :, STATIC_INTRINSIC_DIM:])
        _card_static_board_gpu.copy_(
            torch.from_numpy(_board_np).to(_card_static_board_gpu.dtype),
            non_blocking=True,
        )
        _card_dynamic_gpu.copy_(_card_dynamic_buf, non_blocking=True)
        _hist_info_gpu.copy_(_hist_info_buf, non_blocking=True)
        _shuf_gpu.copy_(_shuf_buf, non_blocking=True)
        _prompt_gpu.copy_(_prompt_buf, non_blocking=True)
        _pb.next_slot()
    else:
        _card_static_board_gpu = _get_partial_gpu(
            "card_static_board", (batch_size, max_cards, STATIC_BOARD_DIM))
        _card_dynamic_gpu = _get_partial_gpu(
            "card_dynamic",    (batch_size, max_cards,     DYNAMIC_CARD_DIM))
        _hist_info_gpu = _get_partial_gpu(
            "hist_info",       (batch_size, max_histories, HISTORY_INFO_DIM))
        _shuf_gpu = _get_partial_gpu("shuffle_indices", (batch_size, max_cards),    torch.int64)
        _prompt_gpu = _get_partial_gpu("prompt_one_hot",  (batch_size, n_prompts))
        _board_np_pb = np.ascontiguousarray(card_static_np[:, :, STATIC_INTRINSIC_DIM:])
        _card_static_board_gpu.copy_(torch.from_numpy(_board_np_pb))
        _card_dynamic_gpu.copy_(_card_dynamic_buf)
        _hist_info_gpu.copy_(_hist_info_buf)
        _shuf_gpu.copy_(_shuf_buf)
        _prompt_gpu.copy_(_prompt_buf)

    card_emb_idx_t = torch.from_numpy(card_emb_idx_np).to(DEVICE)
    hist_emb_idx_t = torch.from_numpy(hist_emb_idx_np).to(DEVICE)
    encoded = EncodedObsCompact(
        card_emb_idx=card_emb_idx_t,
        hist_emb_idx=hist_emb_idx_t,
        card_static_board=_card_static_board_gpu,
        card_dynamic=_card_dynamic_gpu,
        history_info=_hist_info_gpu,
        shuffle_indices=_shuf_gpu,
    )
    return encoded, _prompt_gpu


def encoded_compact_from_ml_numpy(
    obs: dict,
    use_preallocated_gpu: bool = False,
) -> tuple[EncodedObsCompact, torch.Tensor]:
    """Convert native vectorized numpy obs keys into EncodedObsCompact + prompt (on DEVICE)."""
    card_emb_idx_np = np.ascontiguousarray(obs["ml_card_emb_idx_"], dtype=np.int64)
    hist_emb_idx_np = np.ascontiguousarray(obs["ml_hist_emb_idx_"], dtype=np.int64)
    card_static_np = np.ascontiguousarray(obs["ml_card_static_"], dtype=np.float32)
    card_dynamic_np = np.ascontiguousarray(obs["ml_card_dynamic_"], dtype=np.float32)
    history_info_np = np.ascontiguousarray(obs["ml_history_info_"], dtype=np.float32)
    prompt_np = np.ascontiguousarray(obs["ml_prompt_"], dtype=np.float32)
    if card_emb_idx_np.ndim == 1:
        card_emb_idx_np = card_emb_idx_np[None, ...]
        hist_emb_idx_np = hist_emb_idx_np[None, ...]
        card_static_np = card_static_np[None, ...]
        card_dynamic_np = card_dynamic_np[None, ...]
        history_info_np = history_info_np[None, ...]
        prompt_np = prompt_np[None, ...]

    get_lazy_intrinsic_table().update_from_numpy(card_emb_idx_np, card_static_np)
    board_np = np.ascontiguousarray(card_static_np[:, :, STATIC_INTRINSIC_DIM:])
    n_cards = card_emb_idx_np.shape[1]
    shuf = np.full(card_emb_idx_np.shape, -1, dtype=np.int64)
    n_me = obs.get("ml_n_me_")
    if n_me is not None:
        n_me_arr = np.atleast_1d(np.asarray(n_me, dtype=np.int32))
        for b, n in enumerate(n_me_arr):
            n = int(max(0, min(int(n), n_cards)))
            if n:
                shuf[b, :n] = np.arange(n)

    card_emb_idx_t = torch.from_numpy(card_emb_idx_np).to(DEVICE)
    hist_emb_idx_t = torch.from_numpy(hist_emb_idx_np).to(DEVICE)
    encoded = EncodedObsCompact(
        card_emb_idx=card_emb_idx_t,
        hist_emb_idx=hist_emb_idx_t,
        card_static_board=torch.from_numpy(board_np).to(DEVICE),
        card_dynamic=torch.from_numpy(card_dynamic_np).to(DEVICE),
        history_info=torch.from_numpy(history_info_np).to(DEVICE),
        shuffle_indices=torch.from_numpy(shuf).to(DEVICE),
    )
    prompt_t = torch.from_numpy(prompt_np).to(DEVICE)
    return encoded, prompt_t


def load_embeddings(path: Optional[Union[str, Path]] = None) -> dict[str, Union[str, list[float]]]:
    """
    Load card embeddings from a JSON file.

    By default resolves via :func:`ygoenv.paths.embeddings_path`.
    """
    embeddings_file = Path(path) if path is not None else resolve_embeddings_path()
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_file}")
    with embeddings_file.open("r", encoding="utf-8") as f:
        return json.load(f)
