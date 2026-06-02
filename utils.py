import random as rnd
from collections import Counter

def evaluate_hand_counts(hand, community):
    # 1. Combines hand and community cards to create the 7-card pool
    pool = hand + community

    # 2. Extrach rank and suit data
    all_suits = Counter(card[1] for card in pool)
    all_ranks = Counter(card[0] for card in pool)
    unique_ranks = sorted(list(set(all_ranks)), reverse=True)

    rank_counts = Counter(all_ranks)
    suit_counts = Counter(all_suits)

    # 3. Pre-calculate Flush and Straight Status
    has_flush = any(count >= 5 for count in suit_counts.values())
    flush_suit = next((s for s, c in suit_counts.items() if c >= 5), None) if has_flush else None
   
    # Straight Logic
    has_straight = False
    straight_high_card = 0
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i] - unique_ranks[i + 4] == 4:
            has_straight = True
            straight_high_card = unique_ranks[i]
            break
    # Check for Ace-low straight (A-2-3-4-5)
    if {14, 2, 3, 4, 5}.issubset(set(unique_ranks)):
        has_straight = True
        straight_high_card = 5 if straight_high_card < 5 else straight_high_card

    # 4. Straight Flush Logic
    is_straight_flush = False
    sf_high_card = 0
    if has_flush:
        f_ranks = sorted(set(c[0] for c in pool if c[1] == flush_suit), reverse=True)
        for i in range(len(f_ranks) - 4):
            if f_ranks[i] - f_ranks[i + 4] == 4:
                is_straight_flush = True
                sf_high_card = f_ranks[i]
                break
        if {14, 2, 3, 4, 5}.issubset(set(f_ranks)):
            is_straight_flush = True
            sf_high_card = max(sf_high_card, 5)

    #5. Helper for Quads, Trips, and Pairs
    # Sort ranks by frequency first, then by the rank value itself
    sorted_by_freq = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    counts = [item[1] for item in sorted_by_freq]
    freq_ranks = [item[0] for item in sorted_by_freq]
    
    # 6. Hand Evaluation Hierarchy

    # 8. Straight Flush
    if is_straight_flush:
        return (8, sf_high_card)
    
    # 7. Four of a Kind
    elif 4 in counts:
        kicker = max(r for r in all_ranks if r != freq_ranks[0]) # Kicker is highest remaining card
        return (7, freq_ranks[0], kicker)
    
    # 6. Full House
    elif counts[0] == 3 and (len(counts) > 1 and counts[1] >= 2):
        return (6, freq_ranks[0], freq_ranks[1])
    
    # 5. Flush
    elif has_flush:
        f_ranks = sorted([c[0] for c in pool if c[1] == flush_suit], reverse=True)
        return (5, *f_ranks[:5])
    
    # 4. Straight
    elif has_straight:
        return (4, straight_high_card)
    
    # 3. Three of a Kind
    elif counts[0] == 3:
        kickers = sorted([r for r in all_ranks if r != freq_ranks[0]], reverse=True) # Kickers are 2 highest remaining cards
        return (3, freq_ranks[0], *kickers[:2])
    
    # 2. Two Pair
    elif len(counts) >= 2 and counts[1] == 2:
        pair_1, pair_2 = freq_ranks[0], freq_ranks[1]
        kicker = max(r for r in all_ranks if r != pair_1 and r != pair_2) # Kicker is highest remaining card
        return (2, pair_1, pair_2, kicker)
    
    # 1. One Pair
    elif counts[0] == 2:
        pair_rank = freq_ranks[0]
        kickers = sorted([r for r in all_ranks if r != pair_rank], reverse=True) # Kickers are 3 highest remaining cards
        return (1, pair_rank, *kickers[:3])
    
    # 0. High Card
    else:
        return (0, *unique_ranks[:5]) # Kickers are 5 highest cards
    
def format_cards(cards):
        # Maps internal tuple representation (14, 0) to readable string [A♥]
        rank_map = {11: "J", 12: "Q", 13: "K", 14: "A"}
        suit_map = {0: "♥", 1: "♦", 2: "♣", 3: "♠"}
        return " ".join([f"[{rank_map.get(c[0], str(c[0]))}{suit_map[c[1]]}]" for c in cards])

class PokerPlayer:
    def __init__(self, name, initial_stack):
        self.name = name
        self.stack = initial_stack
        self.hand = []
        self.current_bet = 0
        self.is_folded = False
        self.is_all_in = False

    def recieve_cards(self, cards): # Sets the player's hole cards
        self.hand = cards
        self.is_folded = False
        self.is_all_in = False
    
    def place_bet(self, amount):
        if amount >= self.stack:
            actual_bet = self.stack
            self.stack = 0
            self.is_all_in = True
        else:
            actual_bet = amount
            self.stack -= amount
        self.current_bet += actual_bet
        return actual_bet
    
    def fold(self):
        self.is_folded = True
        self.hand = []
    
    def reset_for_new_round(self):
       self.current_bet = 0

    def decide_action(self, highest_bet, to_call, community_cards):
        pass

class PokerDealer:
    def __init__(self, player_names):
        # We define the ranks and suits once
        self.suits = [0, 1, 2, 3] 
        self.ranks = list(range(2, 15)) 
        self.player_names = player_names # Store these for later
        self.community_cards = []
        self.pot = 0
        self.player = {name: [] for name in player_names}
        self.deck = [] # Start empty

    def suffle_and_deal(self): # Note: Matches your current spelling
        # 1. Create a NEW deck of 52 cards every hand
        self.deck = [(r, s) for r in self.ranks for s in self.suits]
        rnd.shuffle(self.deck)
        
        # 2. Reset game state
        self.community_cards = []
        self.pot = 0
        
        # 3. Deal fresh cards
        for name in self.player:
            self.player[name] = [self.deck.pop(), self.deck.pop()]
    
    def deal_flop(self):
        self.deck.pop() # Burn a card  
        self.community_cards.extend([self.deck.pop() for _ in range(3)]) # Deal 3 cards for the flop

    def deal_next_card(self):
      self.deck.pop() # Burn a card
      self.community_cards.append(self.deck.pop()) # Deal 1 card for turn or river
    
    def collect_bets(self, amount):
        self.pot += amount