import numpy as np

def vectorize_state(player, opponent, community_cards, pot_size, highest_bet, is_dealer):
    """
    High-fidelity state representation.
    Vector Length: 52 (Hole) + 52 (Community) + 5 (Metadata) = 109 features
    """    
    # 1. One-hot Encode Cards
    def get_one_hot(cards):
        vec = np.zeros(52)
        for rank, suit_idx in cards:
            # Rank is 2-14, Suit is 0-3
            index = (rank -2) * 4 + suit_idx
            vec[index] = 1
        return vec
    
    hole_cards_vec = get_one_hot(player.hand)
    community_cards_vec = get_one_hot(community_cards)

    # 2. Financial & Positional Metadata
    # Normalized by stack depth (assuming 1000 stack, but we use ratios)
    stack_ratio = player.stack / (player.stack + opponent.stack)
    pot_odds = (highest_bet - player.current_bet) / (pot_size if pot_size > 0 else 1)

    # Is the AI in position? (1 if dealer, 0 if not)
    position = 1.0 if is_dealer else 0.0

    # Street encoding (Pre-flop: 0, Flop: 0.33, Turn: 0.66, River: 1.0)
    street_map = {0: 0.0, 3: 0.33, 4: 0.66, 5: 1.0}
    street_val = street_map.get(len(community_cards), 0.0)

    metadata = np.array([
        stack_ratio,
        pot_odds,
        position,
        street_val,
        highest_bet / 1000.0 # Raw aggression level
    ], dtype=np.float32)

    return np.concatenate([hole_cards_vec, community_cards_vec, metadata])
