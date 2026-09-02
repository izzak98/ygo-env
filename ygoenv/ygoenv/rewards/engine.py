"""
Reward calculation for Yu-Gi-Oh! RL training.

This module evaluates game states against deck-specific reward functions
defined in reward_config.py.
"""

from typing import Protocol, runtime_checkable

from ygoenv.rewards.config import (
    DECK_REWARDS,
    AT_LEAST_ONE_BANISHED,
    AT_LEAST_SEVEN_FROM_HAND_GY_FIELD,
    CardCondition,
    RewardRule,
    Match,
    get_deck_names,
    get_reward_rules,
)


# =============================================================================
# Card Protocol - Define what we expect from card objects
# =============================================================================

@runtime_checkable
class CardLike(Protocol):
    """Protocol defining the expected interface for card objects."""
    card_id: int
    position: str
    location: str
    seq: int
    overlay: bool  # True if card is XYZ material
    negated: bool
    level: int
    race: str | None


# =============================================================================
# Core Matching Logic
# =============================================================================


def _iter_conditions(items) -> list[CardCondition]:
    """Flatten nested condition containers into a plain CardCondition list."""
    if items is None:
        return []
    flattened: list[CardCondition] = []
    stack = [items]
    while stack:
        current = stack.pop()
        if isinstance(current, CardCondition):
            flattened.append(current)
            continue
        if isinstance(current, (list, tuple, set)):
            stack.extend(current)
    return flattened


def _matches_string(actual: str, expected: str, match_type: Match) -> bool:
    """Check if actual string matches expected based on match type."""
    if match_type == Match.FULL:
        return actual == expected
    elif actual is None:
        return False
    else:  # PARTIAL
        return expected in actual


def check_has_materials(cards: list[CardLike], seq: int, location: str) -> bool:
    """Check if the given position and location has any materials."""
    for card in cards:
        if card.seq == seq and card.location == location and card.overlay:
            return True
    return False


def check_condition(cards: list[CardLike], condition: CardCondition) -> int:
    """
    Count how many cards satisfy the given condition.

    Args:
        cards: List of card objects in the game state
        condition: The condition to check against

    Returns:
        Number of cards matching the condition
    """
    count = 0

    for card in cards:
        # Check card ID
        if condition.card_id != "*" and str(card.card_id) != str(condition.card_id):
            continue

        # Check position
        if not _matches_string(card.position, condition.pos.value, condition.pos_match):
            continue

        # Check location
        if not _matches_string(card.location, condition.loc.value, condition.loc_match):
            continue

        # Check overlay restriction
        if condition.not_overlay and card.overlay:
            continue

        if condition.xyz_material and not getattr(card, "overlay", False):
            continue

        if condition.material_for_card_id is not None:
            if not getattr(card, "overlay", False):
                continue
            host_id = str(condition.material_for_card_id)
            seq = getattr(card, "seq", None)
            loc = getattr(card, "location", None)
            attached = False
            for h in cards:
                if getattr(h, "overlay", False):
                    continue
                if str(getattr(h, "card_id", "")) != host_id:
                    continue
                if getattr(h, "seq", None) != seq:
                    continue
                if loc is None or getattr(h, "location", None) != loc:
                    continue
                attached = True
                break
            if not attached:
                continue

        # Check sequence restriction (e.g., Extra Monster Zones 6/7 only)
        if condition.seq_in is not None and card.seq not in condition.seq_in:
            continue

        # Check monster level restriction
        if condition.level_eq is not None and getattr(card, "level", None) != condition.level_eq:
            continue

        # Check race restriction (e.g., Fairy cards in hand)
        if condition.race_eq is not None and str(getattr(card, "race", "")).lower() != condition.race_eq.lower():
            continue

        # YGO type line (Spell/Trap/Monster flags in card.types)
        if condition.type_eq is not None:
            types = getattr(card, "types", None) or []
            want = condition.type_eq.lower()
            if not any(str(t).lower() == want for t in types):
                continue

        if condition.has_material and not check_has_materials(cards, card.seq, card.location):
            continue

        if condition.not_negated and card.negated:
            continue
        count += 1

    return count


def evaluate_rule(cards: list[CardLike], rule: RewardRule) -> float:
    """
    Evaluate a single reward rule against the game state.

    Args:
        cards: List of card objects in the game state
        rule: The reward rule to evaluate

    Returns:
        Reward value (0 if conditions not met)
    """
    target_any_of = getattr(rule, "target_any_of", None)
    if target_any_of:
        counts = [check_condition(cards, c) for c in target_any_of]
        if max(counts, default=0) == 0:
            return 0.0
        # OR across zones: use max so 1 in hand + 1 in GY does not stack to 2 (sum would).
        target_count = max(counts)
    else:
        # Check primary target condition
        target_count = check_condition(cards, rule.target)
        if target_count == 0:
            return 0.0

    min_count = getattr(rule, "min_target_count", None)
    if min_count is not None and target_count < min_count:
        return 0.0

    # Check further conditions if any
    if rule.requires_any:
        # OR logic - at least one must be satisfied
        any_met = any(
            check_condition(cards, cond) > 0
            for cond in _iter_conditions(rule.requires_any)
        )
        if not any_met:
            return 0.0

    elif rule.requires_all:
        # AND logic - all must be satisfied
        all_met = all(
            check_condition(cards, cond) > 0
            for cond in _iter_conditions(rule.requires_all)
        )
        if not all_met:
            return 0.0

    requires_min_combined = getattr(rule, "requires_min_combined_count", None)
    if requires_min_combined is not None:
        conditions, min_total = requires_min_combined
        combined = sum(check_condition(cards, c) for c in conditions)
        if combined < min_total:
            return 0.0

    requires_exact_combined = getattr(rule, "requires_exact_combined_count", None)
    if requires_exact_combined is not None:
        conditions, exact_total = requires_exact_combined
        combined = sum(check_condition(cards, c) for c in conditions)
        if combined != exact_total:
            return 0.0

    # Calculate reward
    if rule.stackable:
        return rule.reward * target_count
    else:
        return rule.reward


def has_at_least_one_banished(cards: list[CardLike]) -> bool:
    """Return True if at least one card is currently banished."""
    return any(check_condition(cards, condition) > 0 for condition in AT_LEAST_ONE_BANISHED)


def has_at_least_seven_hand_gy_monster_spelltrap(cards: list[CardLike]) -> bool:
    """
    Return True if total cards across hand, GY, monster zone, and spell/trap zone >= 7.
    """
    conditions, min_total = AT_LEAST_SEVEN_FROM_HAND_GY_FIELD
    combined_count = sum(check_condition(cards, condition) for condition in conditions)
    return combined_count >= min_total


# =============================================================================
# Main API
# =============================================================================

def get_reward(deck_name: str, cards: list[CardLike]) -> float:
    """
    Calculate total reward for a deck given the current game state.

    Args:
        deck_name: Name of the deck (e.g., "ryzeal", "zoodiac")
        cards: List of card objects representing the game state

    Returns:
        Total reward value

    Raises:
        ValueError: If deck_name is not found
    """
    rules = get_reward_rules(deck_name)

    total_reward = 0.0
    for rule in rules:
        total_reward += evaluate_rule(cards, rule)

    return total_reward


def get_reward_breakdown(deck_name: str, cards: list[CardLike]) -> dict[str, float]:
    """
    Get a detailed breakdown of rewards by rule.

    Useful for debugging and understanding agent behavior.

    Args:
        deck_name: Name of the deck
        cards: List of card objects representing the game state

    Returns:
        Dictionary mapping rule names to their reward contributions
    """
    rules = get_reward_rules(deck_name)

    breakdown = {}
    for rule in rules:
        reward = evaluate_rule(cards, rule)
        if reward > 0:
            breakdown[rule.name] = reward

    return breakdown


# =============================================================================
# Validation Utilities
# =============================================================================

def get_required_card_ids(deck_name: str) -> set[str]:
    """
    Get all card IDs referenced in a deck's reward function.

    Useful for validating that all referenced cards exist in the deck.

    Args:
        deck_name: Name of the deck

    Returns:
        Set of card IDs used in the reward function
    """
    rules = get_reward_rules(deck_name)
    ids: set[str] = set()

    for rule in rules:
        if str(rule.target.card_id).isdigit():
            ids.add(str(rule.target.card_id))

        if rule.requires_any:
            for cond in _iter_conditions(rule.requires_any):
                if str(cond.card_id).isdigit():
                    ids.add(str(cond.card_id))

        if rule.requires_all:
            for cond in _iter_conditions(rule.requires_all):
                if str(cond.card_id).isdigit():
                    ids.add(str(cond.card_id))

        requires_min_combined = getattr(rule, "requires_min_combined_count", None)
        if requires_min_combined is not None:
            conditions, _ = requires_min_combined
            for cond in _iter_conditions(conditions):
                if str(cond.card_id).isdigit():
                    ids.add(str(cond.card_id))
        requires_exact_combined = getattr(rule, "requires_exact_combined_count", None)
        if requires_exact_combined is not None:
            conditions, _ = requires_exact_combined
            for cond in _iter_conditions(conditions):
                if str(cond.card_id).isdigit():
                    ids.add(str(cond.card_id))

    return ids


def check_rewards(deck_name: str, cards: list[CardLike]) -> bool:
    """
    Validate that all card IDs in the reward function exist in the card list.

    Args:
        deck_name: Name of the deck
        cards: List of available cards

    Returns:
        True if all card IDs exist, False otherwise
    """
    required_ids = get_required_card_ids(deck_name)
    available_ids = {str(card.card_id) for card in cards}

    print(f"Card IDs used in reward function for deck {deck_name}:")
    for cid in sorted(required_ids, key=int):
        print(f"  {cid}")

    # Check for missing IDs with bit operations
    missing = required_ids - available_ids

    if missing:
        print("\nThe following card IDs do not exist in the provided card list:")
        for cid in sorted(missing, key=int):
            print(f"  {cid}")
        return False
    else:
        print("\nAll card IDs in the reward function exist in the provided card list.")
        return True


def print_deck_rules(deck_name: str) -> None:
    """Print a human-readable summary of a deck's reward rules."""
    rules = get_reward_rules(deck_name)

    print(f"\n{'='*60}")
    print(f"Reward Rules for: {deck_name}")
    print(f"{'='*60}\n")

    for i, rule in enumerate(rules, 1):
        print(f"{i}. {rule.name}")
        print(f"   Reward: {rule.reward}" + (" (stackable)" if rule.stackable else ""))
        print(
            f"   Target: Card {rule.target.card_id} @ {rule.target.loc.value} ({rule.target.pos.value})")

        if rule.requires_any:
            print(f"   Requires ANY of:")
            for cond in _iter_conditions(rule.requires_any):
                print(f"     - Card {cond.card_id} @ {cond.loc.value}")

        if rule.requires_all:
            print(f"   Requires ALL of:")
            for cond in _iter_conditions(rule.requires_all):
                print(f"     - Card {cond.card_id} @ {cond.loc.value}")
        print()


# =============================================================================
# Legacy Compatibility Layer
# =============================================================================

def read_reward_functions() -> dict:
    """
    Legacy function - returns reward configs in the old JSON format.

    This exists for backwards compatibility. New code should use
    get_reward_rules() directly.
    """
    from reward_config import export_to_json
    return export_to_json()
