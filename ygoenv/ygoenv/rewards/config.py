"""
Reward configuration for Yu-Gi-Oh! RL training.

This module defines reward functions for various decks using a clean, readable
dataclass-based format instead of raw JSON.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =============================================================================
# Core Types
# =============================================================================

class Loc(Enum):
    """Card locations in the game state."""
    MONSTER = "Monster Zone"
    SPELL_TRAP = "Spell & Trap Zone"
    HAND = "Hand"
    GY = "Graveyard"
    BANISHED = "Banished"
    DECK = "Deck"
    EXTRA_DECK = "Extra Deck"


class Pos(Enum):
    """Card positions. Note: Hand cards are 'face-down' unless revealed."""
    FACE_UP = "face-up"
    FACE_DOWN = "face-down"
    FACE_UP_ATTACK = "face-up attack"


class Match(Enum):
    """Matching strategy for location/position checks."""
    FULL = "full"      # Exact string match
    PARTIAL = "partial"  # Substring match (e.g., "Monster" in "Main Monster Zone")


@dataclass
class CardCondition:
    """
    Defines conditions for matching a card in the game state.

    Attributes:
        card_id: The card's database ID
        loc: Required location
        pos: Required position (face-up/face-down)
        loc_match: How to match location (full or partial)
        pos_match: How to match position (full or partial)
        not_overlay: If True, card must NOT be XYZ material
    """
    card_id: str
    loc: Loc
    pos: Pos
    loc_match: Match = Match.PARTIAL
    pos_match: Match = Match.PARTIAL
    not_overlay: bool = False
    xyz_material: bool = False
    not_negated: bool = True
    has_material: bool = False
    seq_in: Optional[tuple[int, ...]] = None
    level_eq: Optional[int] = None
    race_eq: Optional[str] = None
    # YGO type line flag: must appear in card.types (e.g. "Trap", "Spell", "Monster")
    type_eq: Optional[str] = None
    # If set, card must be XYZ material attached to the non-overlay monster with this id (same seq/location)
    material_for_card_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to the legacy JSON format for backwards compatibility."""
        result = {
            "id": str(self.card_id),
            "pos": {
                "type": self.pos.value,
                "match": self.pos_match.value,
            },
            "loc": {
                "type": self.loc.value,
                "match": self.loc_match.value,
            },
        }
        add_conds = [self.not_overlay, self.xyz_material, self.not_negated, self.has_material]
        if (
            any(add_conds)
            or self.seq_in is not None
            or self.level_eq is not None
            or self.race_eq is not None
            or self.type_eq is not None
            or self.material_for_card_id is not None
        ):
            result["further_restrictions"] = {}
            for cond, key in zip(add_conds, ["overlay", "xyz_material", "not_negated", "has_material"]):
                if cond:
                    result["further_restrictions"][key] = cond
            if self.seq_in is not None:
                result["further_restrictions"]["seq_in"] = list(self.seq_in)
            if self.level_eq is not None:
                result["further_restrictions"]["level_eq"] = self.level_eq
            if self.race_eq is not None:
                result["further_restrictions"]["race_eq"] = self.race_eq
            if self.type_eq is not None:
                result["further_restrictions"]["type_eq"] = self.type_eq
            if self.material_for_card_id is not None:
                result["further_restrictions"]["material_for_card_id"] = str(
                    self.material_for_card_id)
        return result


@dataclass
class RewardRule:
    """
    A single reward condition.

    Attributes:
        name: Human-readable description of what this rule rewards
        target: The primary card condition to check
        reward: Reward value when condition is met
        stackable: If True, reward multiplies by count of matching cards
        requires_any: List of conditions where ANY must be met (OR logic)
        requires_all: List of conditions where ALL must be met (AND logic)
    """
    name: str
    target: CardCondition
    reward: float
    stackable: bool = False
    requires_any: Optional[list[CardCondition]] = None
    requires_all: Optional[list[CardCondition]] = None
    # If set, require this many cards matching target (e.g. 2 for "2 I:P")
    min_target_count: Optional[int] = None
    # If set, (conditions, min_total): rule passes only if sum of matches across conditions >= min_total
    requires_min_combined_count: Optional[tuple[list[CardCondition], int]] = None
    # If set, (conditions, exact_total): rule passes only if sum of matches across conditions == exact_total
    requires_exact_combined_count: Optional[tuple[list[CardCondition], int]] = None
    # If set, primary match is ANY of these (OR). `target` is ignored for matching; keep a dummy `target` for dataclass/JSON.
    # Use for "card in hand OR GY" without paying twice when both hold a copy (non-stackable pays once).
    target_any_of: Optional[list[CardCondition]] = None

    def __post_init__(self):
        if self.requires_any and self.requires_all:
            raise ValueError(
                f"Rule '{self.name}' cannot have both requires_any and requires_all. "
                "Use nested rules or restructure the condition."
            )

    @staticmethod
    def _flatten_conditions(items) -> list["CardCondition"]:
        if items is None:
            return []
        flat: list[CardCondition] = []
        stack = [items]
        while stack:
            current = stack.pop()
            if isinstance(current, CardCondition):
                flat.append(current)
                continue
            if isinstance(current, (list, tuple, set)):
                stack.extend(current)
        return flat

    def to_dict(self) -> dict:
        """Convert to the legacy JSON format."""
        result = self.target.to_dict()
        result["name"] = self.name
        result["reward"] = self.reward
        result["stackable"] = self.stackable

        if self.requires_any:
            result["further_conditions"] = {
                "logic": "OR",
                "conditions": [c.to_dict() for c in self._flatten_conditions(self.requires_any)],
            }
        elif self.requires_all:
            result["further_conditions"] = {
                "logic": "AND",
                "conditions": [c.to_dict() for c in self._flatten_conditions(self.requires_all)],
            }
        else:
            result["further_conditions"] = {}

        if self.min_target_count is not None:
            result["min_target_count"] = self.min_target_count

        if self.requires_min_combined_count is not None:
            conditions, min_total = self.requires_min_combined_count
            result["min_combined_count"] = {
                "conditions": [c.to_dict() for c in self._flatten_conditions(conditions)],
                "min": min_total,
            }
        if self.requires_exact_combined_count is not None:
            conditions, exact_total = self.requires_exact_combined_count
            result["exact_combined_count"] = {
                "conditions": [c.to_dict() for c in self._flatten_conditions(conditions)],
                "exact": exact_total,
            }

        if self.target_any_of is not None:
            result["target_any_of"] = [c.to_dict() for c in self.target_any_of]

        return result


# =============================================================================
# Card ID Constants - Organized by Archetype
# =============================================================================

class Cards:
    """
    Named constants for card IDs.
    Much easier to read than magic numbers!
    """

    # -------------------------------------------------------------------------
    # Sky Striker
    # -------------------------------------------------------------------------
    WIDOW_ANCHOR = "98338152"
    SHARK_CANNON = "51227866"

    # -------------------------------------------------------------------------
    # Ryzeal
    # -------------------------------------------------------------------------
    DETONATOR = "34909328"
    CROSS = "6798031"
    DUODRIVE = "7511613"
    PLASMA_HOLE = "33787730"

    # -------------------------------------------------------------------------
    # Zoodiac
    # -------------------------------------------------------------------------
    DRIDENT = "48905153"
    F0_DRACO_FUTURE = "26973555"
    F0_UTOPIC_FUTURE = "65305468"
    RATPIER = "78872731"
    BOARBOW = "74393852"
    HAMMERKONG = "14970113"

    # -------------------------------------------------------------------------
    # Invoked / Dogmatika / Shaddoll
    # -------------------------------------------------------------------------
    MECHABA = "75286621"
    WINDA = "94977269"
    CALIGA = "13529466"
    PUNISHMENT = "82956214"
    NTSS = "80532587"

    # -------------------------------------------------------------------------
    # Tearlament
    # -------------------------------------------------------------------------
    SULLIEK = "74920585"
    KASHTIRA = "4928565"
    SCHEIREN = "572850"
    REINOHEART = "73956664"
    HAVNIS = "37961969"
    MERRLI = "74078255"
    KALEIDO_HEART = "28226490"
    RULKALLOS = "84330567"
    KITKALLOS = "92731385"
    METANOISE = "38436986"
    CRYME = "1329620"

    # -------------------------------------------------------------------------
    # Maliss
    # -------------------------------------------------------------------------
    CRYPTER = "21848500"
    MTP = "94722358"
    RED_RANSOM = "68059897"
    WHITE_BINDER = "95454996"
    GWC = "20726052"
    MARCH_HARE = "20938824"

    # -------------------------------------------------------------------------
    # Link Monsters (Generic)
    # -------------------------------------------------------------------------
    ALLIED_CODE_TALKER = "39138610"
    APOLLOUSA = "4280258"

    # -------------------------------------------------------------------------
    # Bystial
    # -------------------------------------------------------------------------
    DRUISWURM = "6637331"
    MAGNAMHUT = "33854624"
    BALDRAKE = "72656408"

    # -------------------------------------------------------------------------
    # Yubel
    # -------------------------------------------------------------------------
    PHANTOM = "80453041"

    # -------------------------------------------------------------------------
    # Fiendsmith
    # -------------------------------------------------------------------------
    DESIRAE = "82135803"
    SEQUENCE = "49867899"
    REQUIM = "2463794"
    PARADISE = "99989863"
    CRIMSON_LACRIMA = "28803166"
    ENGRAVER = "60764609"
    TRACT = "98567237"
    LURRIE = "97651498"
    NECROQUIP = "93860227"

    # -------------------------------------------------------------------------
    # PK
    # -------------------------------------------------------------------------
    FOG_BLADE = "25542642"

    # -------------------------------------------------------------------------
    # Unchained
    # -------------------------------------------------------------------------
    EXTRA_BLUE_DOG = "67680512"
    CHAMBER = "80801743"

    # -------------------------------------------------------------------------
    # Lunalight
    # -------------------------------------------------------------------------
    LIGER_DANCER = "54701958"

    # -------------------------------------------------------------------------
    # Hero
    # -------------------------------------------------------------------------
    DPE = "60461804"

    # -------------------------------------------------------------------------
    # Plant
    # -------------------------------------------------------------------------
    MUDAN = "71002019"
    SNOW_DROP = "33491462"
    PRIMULA = "8129306"
    PRINCESS = "132308"
    PETAL = "71734607"
    TEARDROP = "33779875"
    STRENNA = "3828844"
    HYPERYTON = "9349094"
    SHEET = "68941332"
    KONKON = "76869711"
    # -------------------------------------------------------------------------
    # Pendulum Magician
    # -------------------------------------------------------------------------
    BLACK_FANG = "75672051"
    XIANGKE = "71692913"
    OAFDRAGON = "14920218"
    STARGAZER = "94415058"
    WISDOM_EYE = "72714461"
    DOULBE_IRIS = "49684352"
    PURPLE_POISION = "48461764"
    HARMONIZING = "7394192"
    TIMEGAZER = "20409757"
    TIME_PENDULUMGRAPH = "1344018"

    # -------------------------------------------------------------------------
    # Red-Eyes
    # -------------------------------------------------------------------------
    DRAGOON = "37818794"

    # -------------------------------------------------------------------------
    # Red Resonator
    # -------------------------------------------------------------------------
    HOT_RED = "9753964"

    # -------------------------------------------------------------------------
    # Odd-Eyes
    # -------------------------------------------------------------------------
    VORTEX = "53262004"

    # -------------------------------------------------------------------------
    # Race
    # -------------------------------------------------------------------------
    CONTAIN = "62777823"
    EXTINGUISH = "99162522"
    PREVENTER = "41443249"
    ARBITRATOR = "2725599"
    AIRLIFTER = "65734501"
    FIRE_ATTACKER = "64612053"
    HYDRANT = "37617348"
    IMPUSLE = "383399996"
    QUICK_ATTACKER = "47425162"
    TURBULENCE = "37495766"

    # -------------------------------------------------------------------------
    # Live Twin
    # -------------------------------------------------------------------------
    KISAKIL = "36326160"
    LILLA = "73810864"
    KI_SIKIL_FROST = "54257392"
    LIL_LA_TREAT = "81078880"
    LIL_LA_SWEET = "82699999"
    CHALLENGE = "98360333"
    EVIL_TWIN_KISAKIL = "9205573"
    EVIL_TWIN_LILLA = "36609518"
    EVIL_TWIN_KI_SIKIL_DEAL = "6636319"
    EVIL_TWIN_DOUBLE_SUNNY = "93672138"

    # -------------------------------------------------------------------------
    # Melo
    # -------------------------------------------------------------------------
    MELO_SCHUBERTA = "57594700"
    MELO_ETOILE = "83793721"

    # -------------------------------------------------------------------------
    # Thunder Dragons
    # -------------------------------------------------------------------------
    TITAN = "41685633"
    COLOSSUS = "15291624"
    DRAGONDARK = "56713174"
    DRAGONMATRIX = "20318029"

    # -------------------------------------------------------------------------
    # Spright
    # -------------------------------------------------------------------------
    RED = "75922381"
    CARROT = "2311090"
    ELF = "27381634"
    SMASHERS = "88836438"

    # -------------------------------------------------------------------------
    # Drytron
    # -------------------------------------------------------------------------
    METEONIS = "69815951"
    HERALD_OF_PERFECTION = "44665365"
    MU_BETA = "1174075"

    # -------------------------------------------------------------------------
    # VV
    # -------------------------------------------------------------------------
    SKULL_GAURDIAN = "10774240"
    SAURAVIS = "4810828"
    LO = "25801745"

    # -------------------------------------------------------------------------
    # Kashira
    # -------------------------------------------------------------------------
    UNICORN = "68304193"
    FENRIR = "32909498"
    ARISE_HEART = "48626373"

    # -------------------------------------------------------------------------
    # Hand Traps / Staples / Generic
    # -------------------------------------------------------------------------
    NIBIRU = "27204311"
    DRAGOSTAPELIA = "69946549"
    TOADALLY_AWESOME = "90809975"
    SP_LITTLE_KNIGHT = "29301450"
    IP_MASK = "65741786"
    VARUDRAS = "70636044"
    CAESAR = "79559912"
    REGULUS = "10604644"
    RHONGO = "63504681"
    GOSSIP_SHADOW = "71166481"
    SAVAGE = "27548199"
    SPHERES = "24361622"
    PROMETHIAN_PRINCESS = "2772337"
    DIABELLSTAR = "72270339"
    LAEVATEINN = "6260560"
    LINKURIBOH = "41999284"
    A_BAO = "4731783"
    DRJINN_BUSTER = "3790062"
    EMP_MEOW_MINE = "48017189"
    MUCKRAKER = "71607202"
    ZOMBIESTEIN = "73445448"
    HOPE_HARBRINGER = "63767246"
    VALON = "40673853"
    SNOW = "55623480"
    DISPATER = "27572350"
    BARONNE = "84815190"
    PHOTON_LORD = "8165596"
    HERALD_OF_THE_ARCHLIGHT = "79606837"
    HERALD_OF_MIRAGE_LIGHTS = "46935289"
    UNDYING_LEGION = "43355214"


# =============================================================================
# Helper Functions for Common Patterns
# =============================================================================


def monster_on_field(card_id: str, not_overlay: bool = True, not_negated: bool = True) -> CardCondition:
    """Shorthand for face-up monster on field."""
    return CardCondition(
        card_id=card_id,
        loc=Loc.MONSTER,
        pos=Pos.FACE_UP,
        not_overlay=not_overlay,
        not_negated=not_negated,
    )


def trap_set(card_id: str) -> CardCondition:
    """Shorthand for face-down card in Spell/Trap zone."""
    return CardCondition(
        card_id=card_id,
        loc=Loc.SPELL_TRAP,
        pos=Pos.FACE_DOWN,
        loc_match=Match.FULL,
        pos_match=Match.FULL,
    )


def spell_face_up(card_id: str) -> CardCondition:
    """Shorthand for face-up continuous/equip spell."""
    return CardCondition(
        card_id=card_id,
        loc=Loc.SPELL_TRAP,
        pos=Pos.FACE_UP,
        loc_match=Match.PARTIAL,
        pos_match=Match.PARTIAL,
    )


def in_hand(card_id: str) -> CardCondition:
    """Shorthand for card in hand (face-down position)."""
    return CardCondition(
        card_id=card_id,
        loc=Loc.HAND,
        pos=Pos.FACE_DOWN,
        loc_match=Match.FULL,
        pos_match=Match.FULL,
    )


def in_gy(card_id: str) -> CardCondition:
    """Shorthand for card in graveyard."""
    return CardCondition(
        card_id=card_id,
        loc=Loc.GY,
        pos=Pos.FACE_UP,
        loc_match=Match.PARTIAL,
        pos_match=Match.PARTIAL,
    )


def in_extra_deck(card_id: str) -> CardCondition:
    """Shorthand for card in Extra Deck (used for 'not yet summoned' checks)."""
    return CardCondition(
        card_id=card_id,
        loc=Loc.EXTRA_DECK,
        pos=Pos.FACE_DOWN,
        loc_match=Match.FULL,
        pos_match=Match.FULL,
    )


def is_xyz_material(card_id: str) -> CardCondition:
    """Shorthand for card being XYZ material."""
    return CardCondition(
        card_id=card_id,
        loc=Loc.MONSTER,
        pos=Pos.FACE_UP,
        xyz_material=True,
    )


def any_card_in_zone(loc: Loc, pos: Pos, loc_match: Match = Match.PARTIAL, pos_match: Match = Match.PARTIAL) -> CardCondition:
    """Shorthand for matching any card in a specific zone/position."""
    return CardCondition(
        card_id="*",
        loc=loc,
        pos=pos,
        loc_match=loc_match,
        pos_match=pos_match,
        not_overlay=False,
        not_negated=False,
    )


def any_monster_on_field() -> list[CardCondition]:
    """
    Any monster in a Monster Zone (face-up or face-down), excluding XYZ materials.

    Returns two conditions (OR); use as requires_any=any_monster_on_field().
    """
    return [
        CardCondition(
            card_id="*",
            loc=Loc.MONSTER,
            pos=Pos.FACE_UP,
            loc_match=Match.PARTIAL,
            pos_match=Match.PARTIAL,
            not_overlay=True,
            not_negated=False,
        ),
        CardCondition(
            card_id="*",
            loc=Loc.MONSTER,
            pos=Pos.FACE_DOWN,
            loc_match=Match.PARTIAL,
            pos_match=Match.PARTIAL,
            not_overlay=True,
            not_negated=False,
        ),
    ]


def any_level_monster_on_field(level: int, not_overlay: bool = True, not_negated: bool = True) -> CardCondition:
    """Shorthand for matching any monster with a specific level on field."""
    return CardCondition(
        card_id="*",
        loc=Loc.MONSTER,
        pos=Pos.FACE_UP,
        not_overlay=not_overlay,
        not_negated=not_negated,
        level_eq=level,
    )


_DRYTRON_CORE_IDS = [
    "14959144",
    "1174075",
    "22420202",
    "60037599",
    "96026108",
    "97148796",
]


def count_card_in_gy(count: int, name: str, partial: bool = True) -> tuple[list[CardCondition], int]:
    """
    Build a combined-count requirement for archetype cards in GY.

    Currently supports archetype key: "drytron".
    """
    key = name.lower().strip()
    if (partial and "drytron" in key) or key == "drytron":
        ids = _DRYTRON_CORE_IDS
    else:
        raise ValueError(f"Unsupported archetype key for count_card_in_gy: {name}")
    return ([in_gy(card_id) for card_id in ids], count)


def count_card_type_in_hand(count: int, race: str) -> tuple[list[CardCondition], int]:
    """Build a combined-count requirement for cards of a specific race in hand."""
    condition = CardCondition(
        card_id="*",
        loc=Loc.HAND,
        pos=Pos.FACE_DOWN,
        loc_match=Match.FULL,
        pos_match=Match.FULL,
        not_overlay=False,
        not_negated=False,
        race_eq=race.strip().title(),
    )
    return ([condition], count)


# YGO main type flags (match decoding.TYPE_STRS entries used in CardRecord.types)
YGO_TYPE_MONSTER = "Monster"
YGO_TYPE_SPELL = "Spell"
YGO_TYPE_TRAP = "Trap"
YGO_TYPE_LINK = "Link"
YGO_TYPE_XYZ = "XYZ"


def any_card_in_hand() -> CardCondition:
    """Wildcard: any card in hand (face-down)."""
    return any_card_in_zone(
        Loc.HAND, Pos.FACE_DOWN, loc_match=Match.FULL, pos_match=Match.FULL
    )


def hand_card_with_ygo_type(card_type: str) -> CardCondition:
    """
    Any hand card whose type line includes the given flag (e.g. Trap, Spell, Monster).

    Use with RewardRule.requires_any / requires_all, or with requires_*_combined_count
    via count_ygo_type_in_hand. Strings match decoding.TYPE_STRS (case-insensitive).
    """
    return CardCondition(
        card_id="*",
        loc=Loc.HAND,
        pos=Pos.FACE_DOWN,
        loc_match=Match.FULL,
        pos_match=Match.FULL,
        not_overlay=False,
        not_negated=False,
        type_eq=card_type.strip(),
    )


def count_ygo_type_in_hand(count: int, card_type: str) -> tuple[list[CardCondition], int]:
    """
    Build requires_min_combined_count / requires_exact_combined_count for YGO type flags in hand.

    Example: count_ygo_type_in_hand(1, YGO_TYPE_TRAP) => at least one trap in hand.
    """
    return ([hand_card_with_ygo_type(card_type)], count)


def plant_xyz_material_under_teardrop() -> CardCondition:
    """XYZ material under Teardrop that is Plant (Rikka lines)."""
    return CardCondition(
        card_id="*",
        loc=Loc.MONSTER,
        pos=Pos.FACE_UP,
        loc_match=Match.PARTIAL,
        pos_match=Match.PARTIAL,
        not_overlay=False,
        not_negated=False,
        xyz_material=True,
        race_eq="Plant",
        type_eq=YGO_TYPE_XYZ,
        material_for_card_id=Cards.TEARDROP,
    )


def any_link_in_spell_trap_zone() -> list[CardCondition]:
    """Any Link monster in a Spell & Trap Zone (face-up or set). Use with requires_any."""
    return [
        CardCondition(
            card_id="*",
            loc=Loc.SPELL_TRAP,
            pos=pos,
            loc_match=Match.PARTIAL,
            pos_match=Match.PARTIAL,
            not_overlay=False,
            not_negated=False,
            type_eq=YGO_TYPE_LINK,
        )
        for pos in (Pos.FACE_UP, Pos.FACE_DOWN)
    ]


AT_LEAST_ONE_BANISHED = [
    any_card_in_zone(Loc.BANISHED, Pos.FACE_UP, loc_match=Match.PARTIAL, pos_match=Match.PARTIAL),
    any_card_in_zone(Loc.BANISHED, Pos.FACE_DOWN, loc_match=Match.PARTIAL, pos_match=Match.PARTIAL),
]


AT_LEAST_SEVEN_FROM_HAND_GY_FIELD = (
    [
        # Hand/GY have a single expected position in this engine.
        any_card_in_zone(Loc.HAND, Pos.FACE_DOWN, loc_match=Match.FULL, pos_match=Match.FULL),
        any_card_in_zone(Loc.GY, Pos.FACE_UP, loc_match=Match.PARTIAL, pos_match=Match.PARTIAL),
        # Field zones can contain both face-up and face-down cards.
        any_card_in_zone(Loc.MONSTER, Pos.FACE_UP,
                         loc_match=Match.PARTIAL, pos_match=Match.PARTIAL),
        any_card_in_zone(Loc.MONSTER, Pos.FACE_DOWN,
                         loc_match=Match.PARTIAL, pos_match=Match.PARTIAL),
        any_card_in_zone(Loc.SPELL_TRAP, Pos.FACE_UP,
                         loc_match=Match.PARTIAL, pos_match=Match.PARTIAL),
        any_card_in_zone(Loc.SPELL_TRAP, Pos.FACE_DOWN,
                         loc_match=Match.PARTIAL, pos_match=Match.PARTIAL),
    ],
    7,
)

# =============================================================================
# Deck Reward Configurations
# =============================================================================


STRIKER_BIKER_REWARDS = [
    RewardRule(
        name="Widow Anchor set",
        target=trap_set(Cards.WIDOW_ANCHOR),
        reward=1.0,
        stackable=True,
    ),
    RewardRule(
        name="Shark Cannon set",
        target=trap_set(Cards.SHARK_CANNON),
        reward=0.9,
        stackable=True,
    ),
]

RYZEAL_REWARDS = [
    RewardRule(
        name="Detonator on field (not as material)",
        target=CardCondition(
            card_id=Cards.DETONATOR,
            loc=Loc.MONSTER,
            pos=Pos.FACE_UP,
            not_overlay=True,
            has_material=True,
        ),
        reward=1.0,
        stackable=True,
    ),
    RewardRule(
        name="Cross on field (with Ryzeal XYZ present)",
        target=CardCondition(
            card_id=Cards.CROSS,
            loc=Loc.SPELL_TRAP,
            pos=Pos.FACE_UP,
            loc_match=Match.PARTIAL,
            pos_match=Match.PARTIAL,
        ),
        reward=1.0,
        stackable=False,
        requires_any=[
            monster_on_field(Cards.DETONATOR, not_overlay=True),
            monster_on_field(Cards.DUODRIVE, not_overlay=True),
        ],
    ),
    RewardRule(
        name="Plasma Hole set (with Ryzeal XYZ present)",
        target=CardCondition(
            card_id=Cards.PLASMA_HOLE,
            loc=Loc.SPELL_TRAP,
            pos=Pos.FACE_DOWN,
            loc_match=Match.PARTIAL,
            pos_match=Match.PARTIAL,
        ),
        reward=1.0,
        stackable=False,
        requires_any=[
            monster_on_field(Cards.DETONATOR, not_overlay=True),
            monster_on_field(Cards.DUODRIVE, not_overlay=True),
        ],
    ),
    RewardRule(
        name="Nibiru in hand (held interaction)",
        target=in_hand(Cards.NIBIRU),
        reward=0.1,
        stackable=True,
    ),
]


ZOODIAC_REWARDS = [
    RewardRule(
        name="Drident on field (not as material)",
        target=CardCondition(
            card_id=Cards.DRIDENT,
            loc=Loc.MONSTER,
            pos=Pos.FACE_UP,
            not_overlay=True,
            has_material=True,
        ),
        reward=1.0,
        stackable=True,
    ),
    RewardRule(
        name="F0 Utopic Draco Future on field (not as material)",
        target=CardCondition(
            card_id=Cards.F0_DRACO_FUTURE,
            loc=Loc.MONSTER,
            pos=Pos.FACE_UP,
            not_overlay=True,
            has_material=True,
        ),
        reward=1.0,
        stackable=True,
    ),
]


INVOKED_REWARDS = [
    RewardRule(
        name="Mechaba on field (monster in hand)",
        target=monster_on_field(Cards.MECHABA),
        reward=0.4,
        stackable=True,
        requires_any=[hand_card_with_ygo_type(YGO_TYPE_MONSTER)],
    ),
    RewardRule(
        name="Mechaba on field (spell in hand)",
        target=monster_on_field(Cards.MECHABA),
        reward=0.4,
        stackable=True,
        requires_any=[hand_card_with_ygo_type(YGO_TYPE_SPELL)],
    ),
    RewardRule(
        name="Mechaba on field (trap in hand)",
        target=monster_on_field(Cards.MECHABA),
        reward=0.4,
        stackable=True,
        requires_any=[hand_card_with_ygo_type(YGO_TYPE_TRAP)],
    ),
    RewardRule(
        name="Winda on field",
        target=monster_on_field(Cards.WINDA),
        reward=0.8,
        stackable=False,
    ),
    RewardRule(
        name="Caliga on field",
        target=monster_on_field(Cards.CALIGA),
        reward=0.8,
        stackable=False,
    ),
    RewardRule(
        name="Punishment set (with N'tss still in Extra)",
        target=trap_set(Cards.PUNISHMENT),
        reward=0.5,
        stackable=True,
        requires_any=[
            in_extra_deck(Cards.NTSS),
        ],
    ),
]


# Tearlaments have a shared condition: any Tear monster on field
_TEAR_MONSTERS_ON_FIELD = [
    monster_on_field(Cards.KASHTIRA, not_overlay=True),
    monster_on_field(Cards.SCHEIREN, not_overlay=True),
    monster_on_field(Cards.REINOHEART, not_overlay=True),
    monster_on_field(Cards.HAVNIS, not_overlay=True),
    monster_on_field(Cards.MERRLI, not_overlay=True),
    monster_on_field(Cards.KALEIDO_HEART, not_overlay=True),
    monster_on_field(Cards.RULKALLOS, not_overlay=True),
    monster_on_field(Cards.KITKALLOS, not_overlay=True),
]

TEAR_REWARDS = [
    RewardRule(
        name="Sulliek set (with any Tear monster on field)",
        target=trap_set(Cards.SULLIEK),
        reward=1.0,
        stackable=False,
        requires_any=_TEAR_MONSTERS_ON_FIELD,
    ),
    RewardRule(
        name="Metanoise set (with any Tear monster on field)",
        target=trap_set(Cards.METANOISE),
        reward=1.0,
        stackable=False,
        requires_any=_TEAR_MONSTERS_ON_FIELD,
    ),
    RewardRule(
        name="Cryme set (with any Tear monster on field and monster in hand)",
        target=trap_set(Cards.CRYME),
        reward=1.0,
        stackable=False,
        requires_any=_TEAR_MONSTERS_ON_FIELD,
        requires_min_combined_count=([hand_card_with_ygo_type(YGO_TYPE_MONSTER)], 1),
    ),
    RewardRule(
        name="Dragostapelia on field",
        target=monster_on_field(Cards.DRAGOSTAPELIA),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Toadally Awesome on field",
        target=monster_on_field(Cards.TOADALLY_AWESOME),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Rulkallos on field",
        target=monster_on_field(Cards.RULKALLOS),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Kaleido-Heart on field (with trap to proc)",
        target=monster_on_field(Cards.KALEIDO_HEART),
        reward=1.0,
        stackable=False,
        requires_any=[
            trap_set(Cards.METANOISE),
            trap_set(Cards.SULLIEK),
            trap_set(Cards.CRYME),
        ],
    ),
    RewardRule(
        name="SP Little Knight on the field",
        target=monster_on_field(Cards.SP_LITTLE_KNIGHT),
        reward=0.5,
        stackable=False,
    ),
]


MALISS_REWARDS = [
    RewardRule(
        name="Allied Code Talker on field",
        target=monster_on_field(Cards.ALLIED_CODE_TALKER),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Apollousa on field",
        target=monster_on_field(Cards.APOLLOUSA),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Crypter on field",
        target=monster_on_field(Cards.CRYPTER),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="MTP set (with Maliss monster present)",
        target=CardCondition(
            card_id=Cards.MTP,
            loc=Loc.SPELL_TRAP,
            pos=Pos.FACE_DOWN,
            loc_match=Match.FULL,
            pos_match=Match.PARTIAL,
        ),
        reward=1.0,
        stackable=False,
        requires_any=[
            monster_on_field(Cards.CRYPTER),
            monster_on_field(Cards.RED_RANSOM),
            monster_on_field(Cards.WHITE_BINDER),
        ],
    ),
    RewardRule(
        name="White Binder in GY (with recursion available)",
        target=in_gy(Cards.WHITE_BINDER),
        reward=1.0,
        stackable=False,
        requires_any=[
            trap_set(Cards.GWC),
            in_hand(Cards.DRUISWURM),
            in_hand(Cards.MAGNAMHUT),
            in_hand(Cards.BALDRAKE),
            in_hand(Cards.MARCH_HARE),
        ],
    ),
]

YUBEL_REWARDS = [
    RewardRule(
        name="Phantom Yubel on field",
        target=monster_on_field(Cards.PHANTOM),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Desirae on field (with sequence or requiem in spell/trap zone)",
        target=monster_on_field(Cards.DESIRAE),
        reward=1.0,
        stackable=False,
        requires_any=[
            spell_face_up(Cards.SEQUENCE),
            spell_face_up(Cards.REQUIM),
        ],
    ),
    RewardRule(
        name="Extra Blue Dog on field (and s:p in extra)",
        target=monster_on_field(Cards.EXTRA_BLUE_DOG),
        reward=0.8,
        stackable=False,
        requires_any=[
            in_extra_deck(Cards.SP_LITTLE_KNIGHT),
        ],
    ),
    RewardRule(
        name="Blue dog in gy (and Chamber in spell/trap zone and s:p in extra)",
        target=in_gy(Cards.EXTRA_BLUE_DOG),
        reward=1,
        stackable=False,
        requires_all=[
            spell_face_up(Cards.CHAMBER),
            in_extra_deck(Cards.SP_LITTLE_KNIGHT),
        ],
    ),
    RewardRule(
        name="Ceasar on field",
        target=monster_on_field(Cards.CAESAR),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Varudras on field",
        target=monster_on_field(Cards.VARUDRAS),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="SP Little Knight on the field",
        target=monster_on_field(Cards.SP_LITTLE_KNIGHT),
        reward=0.5,
        stackable=False,
    ),
]

LUNALIGHT_REWARDS = [
    RewardRule(
        name="Lunalight Liger Dancer on field",
        target=monster_on_field(Cards.LIGER_DANCER),
        reward=1.5,
        stackable=True,
    ),
]

_RIKKA_MONSTERS_ON_FIELD = [
    monster_on_field(Cards.MUDAN),
    monster_on_field(Cards.SNOW_DROP),
    monster_on_field(Cards.PRIMULA),
    monster_on_field(Cards.PRINCESS),
    monster_on_field(Cards.PETAL),
    monster_on_field(Cards.TEARDROP),
    monster_on_field(Cards.STRENNA),
]

PLANT_REWARDS = [
    RewardRule(
        name="Regulus on field",
        target=monster_on_field(Cards.REGULUS),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Teardrop on field (with Plant XYZ material)",
        target=monster_on_field(Cards.TEARDROP),
        reward=1.0,
        stackable=False,
        requires_any=[plant_xyz_material_under_teardrop()],
    ),
    RewardRule(
        name="Princess in hand or GY (with any rikka monster present)",
        target=in_hand(Cards.PRINCESS),
        target_any_of=[in_hand(Cards.PRINCESS), in_gy(Cards.PRINCESS)],
        reward=1.0,
        stackable=False,
        requires_any=_RIKKA_MONSTERS_ON_FIELD,
    ),
    RewardRule(
        name="Sheet set (with any rikka monster present)",
        target=trap_set(Cards.SHEET),
        reward=1.0,
        stackable=False,
        requires_any=_RIKKA_MONSTERS_ON_FIELD,
    ),
    RewardRule(
        name="Konkon on field",
        target=spell_face_up(Cards.KONKON),
        reward=1,
        stackable=False,
        requires_any=[
            in_gy(Cards.PRINCESS),
            in_hand(Cards.PRINCESS),
            trap_set(Cards.SHEET),
        ],
    ),
]

PK_REWARDS = [
    RewardRule(
        name="Fog Blade on field",
        target=trap_set(Cards.FOG_BLADE),
        reward=1.0,
        stackable=True,
    ),
    RewardRule(
        name="Rhongo on field (with Gossip Shadow as Material)",
        target=monster_on_field(Cards.RHONGO),
        reward=5.0,
        stackable=False,
        requires_any=[
            is_xyz_material(Cards.GOSSIP_SHADOW),
        ],
    ),
    RewardRule(
        name="Gossip Shadow in Field",
        target=monster_on_field(Cards.GOSSIP_SHADOW),
        reward=0.8,
        stackable=False,
    ),
    RewardRule(
        name="DPE on field",
        target=monster_on_field(Cards.DPE),
        reward=1.0,
        stackable=False,
    ),
]

_PENDULUM_MAGICIAN_MONSTERS_ON_FIELD = [
    monster_on_field(Cards.BLACK_FANG),
    monster_on_field(Cards.XIANGKE),
    monster_on_field(Cards.OAFDRAGON),
    monster_on_field(Cards.STARGAZER),
    monster_on_field(Cards.WISDOM_EYE),
    monster_on_field(Cards.DOULBE_IRIS),
    monster_on_field(Cards.PURPLE_POISION),
]

_PENDULUM_MAGICIAN_IN_PENDULUM_ZONE = [
    spell_face_up(Cards.BLACK_FANG),
    spell_face_up(Cards.XIANGKE),
    spell_face_up(Cards.OAFDRAGON),
    spell_face_up(Cards.STARGAZER),
    spell_face_up(Cards.WISDOM_EYE),
    spell_face_up(Cards.DOULBE_IRIS),
    spell_face_up(Cards.PURPLE_POISION),
]

_PENDULUM_MAGICIAN_FACE_UP_ANYWHERE = []
_PENDULUM_MAGICIAN_FACE_UP_ANYWHERE.extend(_PENDULUM_MAGICIAN_IN_PENDULUM_ZONE)
_PENDULUM_MAGICIAN_FACE_UP_ANYWHERE.extend(_PENDULUM_MAGICIAN_MONSTERS_ON_FIELD)


PENDULUM_MAGICIAN_REWARDS = [
    RewardRule(
        name="Dragoon on field (with any card in hand)",
        target=monster_on_field(Cards.DRAGOON),
        reward=1.0,
        stackable=False,
        requires_any=[any_card_in_hand()],
    ),
    RewardRule(
        name="Pendulum Graph on field face up (with any pendulum magician monster present)",
        target=spell_face_up(Cards.TIME_PENDULUMGRAPH),
        reward=1.0,
        stackable=False,
        requires_any=_PENDULUM_MAGICIAN_FACE_UP_ANYWHERE,
    ),
    RewardRule(
        name="Pendulum Graph set (with any pendulum magician monster present)",
        target=trap_set(Cards.TIME_PENDULUMGRAPH),
        reward=1.0,
        stackable=False,
        requires_any=_PENDULUM_MAGICIAN_FACE_UP_ANYWHERE,
    ),
    RewardRule(
        name="Vortex on field",
        target=monster_on_field(Cards.VORTEX),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Hot Red on field",
        target=monster_on_field(Cards.HOT_RED),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Savage on field (with any Link in Spell/Trap Zone)",
        target=monster_on_field(Cards.SAVAGE),
        reward=1.0,
        stackable=False,
        requires_any=any_link_in_spell_trap_zone(),
    ),
    RewardRule(
        name="Spheres on field",
        target=CardCondition(
            card_id=Cards.SPHERES,
            loc=Loc.MONSTER,
            pos=Pos.FACE_UP,
            not_overlay=True,
            not_negated=True,
            seq_in=(6, 7),
        ),
        reward=1.0,
        stackable=False,
    ),
]

_RACE_MONSTERS_ON_FIELD = [
    monster_on_field(Cards.PREVENTER, not_negated=False),
    monster_on_field(Cards.ARBITRATOR, not_negated=False),
    monster_on_field(Cards.AIRLIFTER, not_negated=False),
    monster_on_field(Cards.FIRE_ATTACKER, not_negated=False),
    monster_on_field(Cards.HYDRANT, not_negated=False),
    monster_on_field(Cards.QUICK_ATTACKER, not_negated=False),
    monster_on_field(Cards.TURBULENCE, not_negated=False),
]

_RACE_MONSTERS_ON_FIELD_EXCL_PREVENTER = [
    monster_on_field(Cards.ARBITRATOR, not_negated=False),
    monster_on_field(Cards.AIRLIFTER, not_negated=False),
    monster_on_field(Cards.FIRE_ATTACKER, not_negated=False),
    monster_on_field(Cards.QUICK_ATTACKER, not_negated=False),
    monster_on_field(Cards.TURBULENCE, not_negated=False),
    monster_on_field(Cards.HYDRANT, not_negated=False),
]

_RACE_TRAP_SET = [
    trap_set(Cards.CONTAIN),
    trap_set(Cards.EXTINGUISH),
]

_ANY_MONSTER_ON_FIELD_IN_RACE_DECK = [
    monster_on_field(Cards.DIABELLSTAR, not_negated=False),
    monster_on_field(Cards.LAEVATEINN, not_negated=False),
    monster_on_field(Cards.LINKURIBOH, not_negated=False),
    monster_on_field(Cards.PROMETHIAN_PRINCESS, not_negated=False),
    monster_on_field(Cards.SP_LITTLE_KNIGHT, not_negated=False),
    monster_on_field(Cards.IP_MASK, not_negated=False),
]
_ANY_MONSTER_ON_FIELD_IN_RACE_DECK.extend(_RACE_MONSTERS_ON_FIELD)

_ANY_FIRE_MONSTER_ON_FIELD_IN_RACE_DECK = [
    monster_on_field(Cards.LAEVATEINN, not_negated=False),
    monster_on_field(Cards.PROMETHIAN_PRINCESS, not_negated=False),
]
_ANY_FIRE_MONSTER_ON_FIELD_IN_RACE_DECK.extend(_RACE_MONSTERS_ON_FIELD)

RACE_REWARDS = [
    RewardRule(
        name="Containt Set",
        target=trap_set(Cards.CONTAIN),
        reward=0.2,
        stackable=False,
    ),
    RewardRule(
        name="Extinguish Set",
        target=trap_set(Cards.EXTINGUISH),
        reward=0.2,
        stackable=False,
    ),
    RewardRule(
        name="Containt Set (with any race monster present)",
        target=trap_set(Cards.CONTAIN),
        reward=0.8,
        stackable=False,
        requires_any=_RACE_MONSTERS_ON_FIELD_EXCL_PREVENTER,
    ),
    RewardRule(
        name="Extinguish Set (with any race monster present)",
        target=trap_set(Cards.EXTINGUISH),
        reward=0.8,
        stackable=False,
        requires_any=_RACE_MONSTERS_ON_FIELD_EXCL_PREVENTER,
    ),
    RewardRule(
        name="Preventer on field (with any race monster present)",
        target=monster_on_field(Cards.PREVENTER),
        reward=1.0,
        stackable=False,
        requires_any=_RACE_MONSTERS_ON_FIELD_EXCL_PREVENTER,
    ),
    RewardRule(
        name="Arbitrator on field (with any race trap set present)",
        target=monster_on_field(Cards.ARBITRATOR),
        reward=1.0,
        stackable=False,
        requires_any=_RACE_TRAP_SET,
    ),
    RewardRule(
        name="S:P on field",
        target=monster_on_field(Cards.SP_LITTLE_KNIGHT),
        reward=0.5,
        stackable=False,
    ),
    RewardRule(
        name="I:P + any other monster on field with S:P in extra",
        target=monster_on_field(Cards.IP_MASK),
        reward=1.0,
        stackable=False,
        requires_all=[in_extra_deck(Cards.SP_LITTLE_KNIGHT)],
        requires_min_combined_count=(_ANY_MONSTER_ON_FIELD_IN_RACE_DECK, 2),
    ),
    RewardRule(
        name="Promethian Princess in Gy (with any fire monster present)",
        target=in_gy(Cards.PROMETHIAN_PRINCESS),
        reward=1.0,
        stackable=False,
        requires_any=_ANY_FIRE_MONSTER_ON_FIELD_IN_RACE_DECK,
    ),
]

_ANY_LINK_2_IN_LIVE_TWIN = [
    monster_on_field(Cards.EVIL_TWIN_KISAKIL, not_negated=False),
    monster_on_field(Cards.EVIL_TWIN_LILLA, not_negated=False),
    monster_on_field(Cards.EVIL_TWIN_DOUBLE_SUNNY, not_negated=False),
    monster_on_field(Cards.SEQUENCE, not_negated=False),
    monster_on_field(Cards.MUCKRAKER, not_negated=False),
    monster_on_field(Cards.SP_LITTLE_KNIGHT, not_negated=False),
]

_KISA_KIL_OR_LIL_LA = [
    monster_on_field(Cards.KI_SIKIL_FROST, not_negated=False),
    monster_on_field(Cards.LIL_LA_TREAT, not_negated=False),
    monster_on_field(Cards.LIL_LA_SWEET, not_negated=False),
    monster_on_field(Cards.KI_SIKIL_FROST, not_negated=False),
]

_KI_SIKIL = [
    in_gy(Cards.KI_SIKIL_FROST),
    in_gy(Cards.EVIL_TWIN_KI_SIKIL_DEAL),
]

LIVE_TWIN_REWARDS = [
    RewardRule(
        name="EMP in GY (with any link 2 present)",
        target=in_gy(Cards.EMP_MEOW_MINE),
        reward=0.4,
        stackable=False,
        requires_any=_ANY_LINK_2_IN_LIVE_TWIN,
    ),
    RewardRule(
        name="Challenge Set (with any ki-sikil or lil-la present)",
        target=trap_set(Cards.CHALLENGE),
        reward=0.4,
        stackable=False,
        requires_any=_KISA_KIL_OR_LIL_LA,
    ),
    RewardRule(
        name="Crimson Lacrima in GY (with desire in extra)",
        target=in_gy(Cards.CRIMSON_LACRIMA),
        reward=0.7,
        stackable=False,
        requires_any=[in_extra_deck(Cards.DESIRAE)],
    ),
    RewardRule(
        name="Paradise in gy (with desire in extra)",
        target=in_gy(Cards.PARADISE),
        reward=0.6,
        stackable=False,
        requires_any=[in_extra_deck(Cards.DESIRAE)],
    ),
    RewardRule(
        name="Paradise set (with desire on field)",
        target=trap_set(Cards.PARADISE),
        reward=0.6,
        stackable=False,
        requires_any=[monster_on_field(Cards.DESIRAE, not_negated=False)],
    ),
    RewardRule(
        name="Desire on field (with req of either seq or requiem in spell/trap zone)",
        target=monster_on_field(Cards.DESIRAE),
        reward=1.0,
        stackable=False,
        requires_any=[
            spell_face_up(Cards.SEQUENCE),
            spell_face_up(Cards.REQUIM),
        ],
    ),
    RewardRule(
        name="Ceaser on field",
        target=monster_on_field(Cards.CAESAR),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="A-Bao on field (with any card in hand)",
        target=monster_on_field(Cards.A_BAO),
        reward=1.0,
        stackable=False,
        requires_any=[any_card_in_hand()],
    ),
    RewardRule(
        name="Drjinn Buster on field",
        target=monster_on_field(Cards.DRJINN_BUSTER),
        reward=0.5,
        stackable=False,
    ),
    RewardRule(
        name="Evil Twin Kisakil on field (with Evil Twin Lil-la in GY)",
        target=monster_on_field(Cards.EVIL_TWIN_KISAKIL),
        reward=1.0,
        stackable=False,
        requires_any=[in_gy(Cards.EVIL_TWIN_LILLA)],
    ),
    RewardRule(
        name="SP on field",
        target=monster_on_field(Cards.SP_LITTLE_KNIGHT),
        reward=0.5,
        stackable=False,
    ),
    RewardRule(
        name="Troubly Sunny (with Evil Twin Lil-la in GY and any ki-sikil in GY)",
        target=monster_on_field(Cards.EVIL_TWIN_DOUBLE_SUNNY),
        reward=1.0,
        stackable=False,
        requires_any=[in_gy(Cards.EVIL_TWIN_LILLA)],
        requires_min_combined_count=(_KI_SIKIL, 1),
    ),
]

MELO_REWARDS = [
    RewardRule(
        name="Melo Schuberta on field",
        target=monster_on_field(Cards.MELO_SCHUBERTA),
        reward=0.7,
        stackable=True,
    ),
    RewardRule(
        name="Melo Etoile on field",
        target=monster_on_field(Cards.MELO_ETOILE),
        reward=1.0,
        stackable=False,
    ),
]

THUNDER_DRAGONS_REWARDS = [
    RewardRule(
        name="Titan on field (with dragondark in hand)",
        target=monster_on_field(Cards.TITAN),
        reward=0.7,
        stackable=False,
        requires_any=[in_hand(Cards.DRAGONDARK)],
    ),
    RewardRule(
        name="Titan on field (with dragonmatrix in hand)",
        target=monster_on_field(Cards.TITAN),
        reward=0.7,
        stackable=False,
        requires_any=[in_hand(Cards.DRAGONMATRIX)],
    ),
    RewardRule(
        name="Colossus on field",
        target=monster_on_field(Cards.COLOSSUS),
        reward=0.9,
        stackable=False,
    ),
    RewardRule(
        name="Zombiestein on field (in ATK position)",
        target=CardCondition(
            card_id=Cards.ZOMBIESTEIN,
            loc=Loc.MONSTER,
            pos=Pos.FACE_UP_ATTACK,
            pos_match=Match.FULL,
            not_overlay=True,
            not_negated=True,
        ),
        reward=1.0,
        stackable=False,
        requires_any=[monster_on_field(Cards.ZOMBIESTEIN)],
    ),
    RewardRule(
        name="Hope Harbringer on field",
        target=monster_on_field(Cards.HOPE_HARBRINGER),
        reward=0.6,
        stackable=False,
        requires_any=[monster_on_field(Cards.HOPE_HARBRINGER)],
    ),
    RewardRule(
        name="Valon on field",
        target=monster_on_field(Cards.VALON),
        reward=1.0,
        stackable=False,
    ),
]

LIGTHSWORN_REWARDS = [
    RewardRule(
        name="Snow in gy (with access to at least 7 cards)",
        target=in_gy(Cards.SNOW),
        reward=0.3,
        stackable=False,
        requires_any=AT_LEAST_SEVEN_FROM_HAND_GY_FIELD,
    ),
    RewardRule(
        name="Dispaters on field (with at least one banished card)",
        target=monster_on_field(Cards.DISPATER),
        reward=1.0,
        stackable=False,
        requires_any=AT_LEAST_ONE_BANISHED,
    ),
    RewardRule(
        name="Baronne on field",
        target=monster_on_field(Cards.BARONNE),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Photon Lord on field",
        target=monster_on_field(Cards.PHOTON_LORD),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Apollousa on field",
        target=monster_on_field(Cards.APOLLOUSA),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Spheres on field",
        target=CardCondition(
            card_id=Cards.SPHERES,
            loc=Loc.MONSTER,
            pos=Pos.FACE_UP,
            not_overlay=True,
            not_negated=True,
            seq_in=(6, 7),
        ),
        reward=1.0,
        stackable=False,
    ),
]

SPRIGHT_REWARDS = [
    RewardRule(
        name="Red on field",
        target=monster_on_field(Cards.RED),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Carrot on field",
        target=monster_on_field(Cards.CARROT),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Toadally Awesome on field",
        target=monster_on_field(Cards.TOADALLY_AWESOME),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Djinn Buster on field",
        target=monster_on_field(Cards.DRJINN_BUSTER),
        reward=0.5,
        stackable=False,
    ),
    RewardRule(
        name="Baronne on field",
        target=monster_on_field(Cards.BARONNE),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Smashers set (with any level 2 monster present)",
        target=trap_set(Cards.SMASHERS),
        reward=1.0,
        stackable=False,
        requires_any=[any_level_monster_on_field(2, not_overlay=True, not_negated=False)],
    ),
]

DRYTRON_REWARDS = [
    RewardRule(
        name="Meteonis on field (with any 2 drytron in gy)",
        target=monster_on_field(Cards.METEONIS),
        reward=0.8,
        stackable=False,
        requires_min_combined_count=count_card_in_gy(2, name="drytron", partial=True),
    ),
    RewardRule(
        name="Herald of the Archlight on field ",
        target=monster_on_field(Cards.HERALD_OF_THE_ARCHLIGHT),
        reward=0.8,
        stackable=False,
    ),
    RewardRule(
        name="Herald of the Mirage Lights on field (with at least one fairy in hand)",
        target=monster_on_field(Cards.HERALD_OF_MIRAGE_LIGHTS),
        reward=0.8,
        stackable=False,
        requires_min_combined_count=count_card_type_in_hand(1, "fairy"),
    ),
    RewardRule(
        name="Mu Beta on field (with meteonis also on field)",
        target=monster_on_field(Cards.MU_BETA),
        reward=0.6,
        stackable=False,
        requires_any=[monster_on_field(Cards.METEONIS)],
    ),
    *[RewardRule(
        name=f"Herald of Perfection on field (with {n} fairies in hand)",
        target=monster_on_field(Cards.HERALD_OF_PERFECTION),
        reward=1.0,
        stackable=False,
        requires_exact_combined_count=count_card_type_in_hand(n, "fairy"),
    ) for n in range(1, 10)],
]

_SUB_RIKKA_MONSTERS_ON_FIELD = [
    monster_on_field(Cards.PRINCESS),
    monster_on_field(Cards.STRENNA),
    monster_on_field(Cards.TEARDROP),
]


SYLVANS = [
    RewardRule(
        name="Baronne on field",
        target=monster_on_field(Cards.BARONNE),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Savage on field (with any Link in Spell/Trap Zone)",
        target=monster_on_field(Cards.SAVAGE),
        reward=1.0,
        stackable=False,
        requires_any=any_link_in_spell_trap_zone(),
    ),
    RewardRule(
        name="Teardrop the Rikka Queen on field (with Plant XYZ material)",
        target=monster_on_field(Cards.TEARDROP),
        reward=1.0,
        stackable=False,
        requires_any=[plant_xyz_material_under_teardrop()],
    ),
    RewardRule(
        name="Appolousa on field",
        target=monster_on_field(Cards.APOLLOUSA),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Princess in hand or GY (with any Rikka monster present)",
        target=in_hand(Cards.PRINCESS),
        target_any_of=[in_hand(Cards.PRINCESS), in_gy(Cards.PRINCESS)],
        reward=1.0,
        stackable=False,
        requires_any=_SUB_RIKKA_MONSTERS_ON_FIELD,
    ),
    RewardRule(
        name="Heard of Archlight on field",
        target=monster_on_field(Cards.HERALD_OF_THE_ARCHLIGHT),
        reward=1.0,
        stackable=False,
    ),
]

VOICELESS_REWARDS = [
    RewardRule(
        name="Skull Guardian on field (with Lo on field)",
        target=monster_on_field(Cards.SKULL_GAURDIAN),
        reward=1.0,
        stackable=False,
        requires_any=[monster_on_field(Cards.LO)],
    ),
    RewardRule(
        name="Sauravis on field",
        target=monster_on_field(Cards.SAURAVIS),
        reward=1.0,
        stackable=False,
    ),
    RewardRule(
        name="Sauravis in hand (with any monster on field)",
        target=in_hand(Cards.SAURAVIS),
        reward=0.1,
        stackable=False,
        requires_any=any_monster_on_field(),
    ),
    RewardRule(
        name="Unicorn on field",
        target=monster_on_field(Cards.UNICORN),
        reward=0.3,
        stackable=False,
    ),
    RewardRule(
        name="Fenrir on field",
        target=monster_on_field(Cards.FENRIR),
        reward=0.6,
        stackable=False,
    ),
    RewardRule(
        name="Undying Legion on field (with xuz material)",
        target=CardCondition(
            card_id=Cards.UNDYING_LEGION,
            loc=Loc.MONSTER,
            pos=Pos.FACE_UP,
            not_overlay=True,
            has_material=True,
        ),
        reward=0.7,
        stackable=True,
    ),
    RewardRule(
        name="Arise Heart on field (with xyz material)",
        target=CardCondition(
            card_id=Cards.ARISE_HEART,
            loc=Loc.MONSTER,
            pos=Pos.FACE_UP,
            not_overlay=True,
            has_material=True,
        ),
        reward=1.0,
        stackable=True,
    ),
]

# =============================================================================
# Deck Registry
# =============================================================================

DECK_REWARDS: dict[str, list[RewardRule]] = {
    "striker_biker": STRIKER_BIKER_REWARDS,
    "ryzeal": RYZEAL_REWARDS,
    "zoodiac": ZOODIAC_REWARDS,
    "invoked": INVOKED_REWARDS,
    "tear": TEAR_REWARDS,
    "maliss": MALISS_REWARDS,
    "yubel": YUBEL_REWARDS,
    "lunalight": LUNALIGHT_REWARDS,
    "plants": PLANT_REWARDS,
    "pk": PK_REWARDS,
    "pend_magician": PENDULUM_MAGICIAN_REWARDS,
    "race": RACE_REWARDS,
    "live_twin": LIVE_TWIN_REWARDS,
    "melo": MELO_REWARDS,
    "t_drag": THUNDER_DRAGONS_REWARDS,
    "ligthsworn": LIGTHSWORN_REWARDS,
    "spright": SPRIGHT_REWARDS,
    "drytron": DRYTRON_REWARDS,
    "sylvans": SYLVANS,
    "vv": VOICELESS_REWARDS,
}


def get_deck_names() -> list[str]:
    """Return all available deck names."""
    return list(DECK_REWARDS.keys())


def get_reward_rules(deck_name: str) -> list[RewardRule]:
    """Get reward rules for a specific deck."""
    if deck_name not in DECK_REWARDS:
        raise ValueError(
            f"Unknown deck: {deck_name}. Available: {get_deck_names()}"
        )
    return DECK_REWARDS[deck_name]


def export_to_json() -> dict:
    """
    Export all deck configs to the legacy JSON format.
    Useful for backwards compatibility or debugging.
    """
    return {
        deck_name: [rule.to_dict() for rule in rules]
        for deck_name, rules in DECK_REWARDS.items()
    }
