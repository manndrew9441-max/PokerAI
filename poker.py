import torch
import numpy as np
from collections import Counter
from PokerAI import SmartAIPlayer
from utils import evaluate_hand_counts, PokerPlayer, PokerDealer, format_cards
from ReplayBuffer import ReplayBuffer

"""
Human Player Logic
"""
class HumanPlayer(PokerPlayer):
   def decide_action(self, highest_bet, to_call, community_cards, pot_size=0, opponent=None, is_dealer=False):
        print(f"\n{self.name}'s turn. Stack: {self.stack}")
        
        if to_call == 0:
            user_input = input("Action (Check/Bet [amount]): ").lower().strip()
            if not user_input or user_input == "check":
                return "check"
        else:
            print(f"Current bet to call: {to_call}")
            user_input = input("Action (Fold/Call/Raise [amount]): ").lower().strip()
            if user_input == "fold": return "fold"
            if user_input == "call": return "call"

        # 1. Split the input into parts (e.g., "raise 50" -> ["raise", "50"])
        parts = user_input.split()
        
        # 2. Find the number in the input
        amount = None
        for part in parts:
            if part.isdigit():
                amount = int(part)
                break
        
        # 3. If a number was found, calculate the TOTAL target for the engine
        if amount is not None:
            # We add what we've already bet to the new amount
            total_target = self.current_bet + int(amount)
            return f"raise {total_target}"
            
        # 4. Fallback if no number was found
        return "check" if to_call == 0 else "fold"

"""
Game Controller Logic
"""
class TexasHoldemGame:
    def __init__(self, human_name, ai_name, starting_stack):
        self.dealer_engine = PokerDealer([human_name, ai_name])
        self.players = [
            HumanPlayer(human_name, starting_stack),
            SmartAIPlayer(ai_name, starting_stack)
        ]
        self.sb_amount = 10
        self.bb_amount = 20
        self.button_index = 0  # 0: Human is Dealer, 1: AI is Dealer

    def play_round(self):
        # 1. Setup Phase
        self.dealer_engine.suffle_and_deal()
        sb_idx = self.button_index
        bb_idx = (self.button_index + 1) % 2
        
        sb_p = self.players[sb_idx]
        bb_p = self.players[bb_idx]

        for p in self.players:
            p.recieve_cards(self.dealer_engine.player[p.name])
            p.reset_for_new_round()
        
        print(f"\n--- NEW ROUND (Dealer: {self.players[sb_idx].name}) ---")
        print(f"Your Hand: {format_cards(self.players[0].hand)}")

        # 2. Pre-Flop: Post Blinds
        self.dealer_engine.collect_bets(sb_p.place_bet(self.sb_amount))
        self.dealer_engine.collect_bets(bb_p.place_bet(self.bb_amount))
        print(f"Blinds: {sb_p.name} (SB) {self.sb_amount}, {bb_p.name} (BB) {self.bb_amount}")
        
        # In Heads-up Pre-Flop, Dealer/SB acts first
        self.run_betting_phase("Pre-Flop", initial_highest=self.bb_amount, start_idx=sb_idx)

        # 3. The Flop
        if self.can_continue():
            self.dealer_engine.deal_flop()
            print(f"\n--- FLOP --- {format_cards(self.dealer_engine.community_cards)}")
            print(f"Your Hand: {format_cards(self.players[0].hand)}")
            # Post-Flop, Big Blind (out of position) acts first
            self.run_betting_phase("Flop", start_idx=bb_idx)

        # 4. The Turn
        if self.can_continue():
            self.dealer_engine.deal_next_card()
            print(f"\n--- TURN --- {format_cards(self.dealer_engine.community_cards)}")
            print(f"Your Hand: {format_cards(self.players[0].hand)}")
            self.run_betting_phase("Turn", start_idx=bb_idx)

        # 5. The River
        if self.can_continue():
            self.dealer_engine.deal_next_card()
            print(f"\n--- RIVER --- {format_cards(self.dealer_engine.community_cards)}")
            print(f"Your Hand: {format_cards(self.players[0].hand)}")
            self.run_betting_phase("River", start_idx=bb_idx)

        # 6. Showdown
        self.handle_showdown()
        
        # Rotate Dealer Button for the next hand
        self.button_index = (self.button_index + 1) % 2

    def run_betting_phase(self, phase_name, initial_highest=0, start_idx=0):
        highest_bet = initial_highest
        players_acted = {p.name: False for p in self.players}

        if phase_name != "Pre-Flop":
            for p in self.players: 
                p.reset_for_new_round()
            highest_bet = 0

        while True:
            active_players = [p for p in self.players if not p.is_folded and not p.is_all_in]
            
            # Check if betting is finished before starting the turn rotation
            if all(p.current_bet == highest_bet for p in active_players) and \
               all(players_acted[p.name] for p in active_players):
                break
            
            for i in range(len(self.players)):
                curr_idx = (start_idx + i) % len(self.players)
                p = self.players[curr_idx] # <--- 'p' is defined here

                if p.is_folded or p.is_all_in:
                    continue
                
                if p.current_bet == highest_bet and players_acted[p.name]:
                    continue

                to_call = highest_bet - p.current_bet
                
                # THE CALL MUST BE INSIDE THIS FOR-LOOP
                # Determine who the opponent is for the AI
                opponent = self.players[0] if p == self.players[1] else self.players[1]
                is_dealer = (self.button_index == self.players.index(p))

                action = p.decide_action(
                    highest_bet, 
                    to_call, 
                    self.dealer_engine.community_cards, 
                    self.dealer_engine.pot, 
                    opponent, 
                    is_dealer
                )
                players_acted[p.name] = True

                if action == "fold":
                    p.fold()
                    print(f"{p.name} folds.")
                elif action in ["call", "check"]:
                    self.dealer_engine.collect_bets(p.place_bet(to_call))
                    print(f"{p.name} {action}s.")
                elif action.startswith("raise"):
                    try:
                        parts = action.split()
                        new_total_bet = int(parts[1])
                        
                        # NEW: If the player types exactly what is needed to call 
                        # (e.g., they have 10 in, they type 20), treat it as a call.
                        if new_total_bet == highest_bet:
                            to_call = highest_bet - p.current_bet
                            self.dealer_engine.collect_bets(p.place_bet(to_call))
                            print(f"{p.name} calls {highest_bet}.")
                        
                        # If the amount is higher, it's a valid raise
                        elif new_total_bet > highest_bet:
                            diff = new_total_bet - p.current_bet
                            highest_bet = new_total_bet 
                            self.dealer_engine.collect_bets(p.place_bet(diff))
                            
                            verb = "bet" if (highest_bet - diff == 0) else "raised to"
                            print(f"{p.name} {verb} {highest_bet}!")
                            
                            for name in players_acted:
                                if name != p.name:
                                    players_acted[name] = False
                        
                        # Otherwise, they tried to bet less than the current price
                        else:
                            to_call = highest_bet - p.current_bet
                            self.dealer_engine.collect_bets(p.place_bet(to_call))
                            print(f"{p.name} tried to raise too low. Calling {highest_bet} instead.")

                    except (ValueError, IndexError):
                        # Handle bad text inputs
                        to_call = highest_bet - p.current_bet
                        self.dealer_engine.collect_bets(p.place_bet(to_call))
                        print(f"{p.name} provided an invalid input. Calling {highest_bet} instead.")

    def can_continue(self):
        """Checks if at least two players are still in the hand."""
        return len([p for p in self.players if not p.is_folded]) > 1

    def handle_showdown(self):
        active = [p for p in self.players if not p.is_folded]
        
        # Scenario: Everyone but one person folded
        if len(active) == 1:
            winner = active[0]
            print(f"\n{winner.name} wins the pot of {self.dealer_engine.pot}!")
            winner.stack += self.dealer_engine.pot
            self.dealer_engine.pot = 0
            return

        print("\n--- SHOWDOWN ---")
        
        # Translation Helper for rank formatting
        rank_names = {11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}
        def get_name(r): return rank_names.get(r, str(r))

        results = []
        for p in active:
            # FIX: Using self.dealer_engine instead of self.dealer
            score = evaluate_hand_counts(p.hand, self.dealer_engine.community_cards)
            results.append((p, score))
            
            # Detailed hand reporting with kicker translation
            hand_rank = score[0]
            primary = get_name(score[1])
            
            if hand_rank == 8: desc = f"Straight Flush, {primary} high"
            elif hand_rank == 7: desc = f"Four of a Kind, {primary}s with {get_name(score[2])} kicker"
            elif hand_rank == 6: desc = f"Full House, {primary}s full of {get_name(score[2])}s"
            elif hand_rank == 5: desc = f"Flush, {', '.join([get_name(k) for k in score[1:6]])}"
            elif hand_rank == 4: desc = f"Straight, {primary} high"
            elif hand_rank == 3: desc = f"Three of a Kind, {primary}s with {', '.join([get_name(k) for k in score[2:4]])} kickers"
            elif hand_rank == 2: desc = f"Two Pair, {primary}s and {get_name(score[2])}s with {get_name(score[3])} kicker"
            elif hand_rank == 1: desc = f"Pair of {primary}s with {', '.join([get_name(k) for k in score[2:5]])} kickers"
            else: desc = f"High Card {primary} with {', '.join([get_name(k) for k in score[2:6]])} kickers"

            print(f"{p.name}: {format_cards(p.hand)} | {desc}")

        # 1. Sort results by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 2. Find the highest score achieved
        best_score = results[0][1]
        
        # 3. Identify all players who tied for that best score
        winners = [p for p, score in results if score == best_score]
        
        # 4. Divide the pot
        split_amount = self.dealer_engine.pot // len(winners)
        
        if len(winners) > 1:
            print(f"\nIt's a TIE! Split pot between: {', '.join([w.name for w in winners])}")
            for w in winners:
                print(f"{w.name} receives {split_amount} chips.")
                w.stack += split_amount
                winners[0].stack += self.dealer_engine.pot % len(winners)
        else:
            winner = winners[0]
            print(f"\n{winner.name} wins the pot of {self.dealer_engine.pot}!")
            winner.stack += self.dealer_engine.pot
            
        # 5. Reset pot and handle any leftover "dust" chips from odd divisions
        self.dealer_engine.pot = 0

    def finalize_rewards(self, initial_stacks, replay_buffer): # For AI
        # We must loop through players to find the SmartAIPlayer
        for p in self.players:
            if isinstance(p, SmartAIPlayer) and len(p.memory) > 0:
                # Calculate the total win/loss for the hand
                reward = float(p.stack - initial_stacks[p.name])
                
                for i in range(len(p.memory)):
                    state = p.memory[i]['state']
                    action = p.memory[i]['action']
                    
                    # Determine next_state for the Markov Chain
                    if i + 1 < len(p.memory):
                        next_state = p.memory[i+1]["state"]
                        done = 0
                    else:
                        # End of hand terminal state
                        next_state = np.zeros(109, dtype=np.float32)
                        done = 1
                    
                    replay_buffer.push(state, action, reward, next_state, done)
                
                # Clear memory for next hand and update exploration rate
                p.memory = []
                if p.epsilon > p.epsilon_min:
                    p.epsilon *= p.epsilon_decay

"""
Main Game Loop
"""
if __name__ == "__main__":
    # 1. Initialize the global buffer
    global_replay_buffer = ReplayBuffer(capacity=50000)
    
    game = TexasHoldemGame("Player", "PokerAI", 1000)

    # Load trained weights into the AI
    ai_player = game.players[1]  # SmartAIPlayer is index 1
    checkpoint = torch.load("checkpoints/AI_1_final_hand_1000000.pth")
    ai_player.model.load_state_dict(checkpoint['model_state'])
    ai_player.epsilon = 0.0  # No random moves — pure learned strategy
    ai_player.model.eval()

    print("Trained model loaded. Good luck!")
        
    playing = True
    while playing:
        # 2. Record stacks BEFORE the hand starts
        initial_stacks = {p.name: p.stack for p in game.players}
        
        game.play_round()

        # 3. Finalize rewards using the captured initial stacks
        game.finalize_rewards(initial_stacks, global_replay_buffer)

        # Check for bankruptcy
        for p in game.players:
            if p.stack <= 0:
                print(f"\n{p.name} is out of chips! Game over.")
                playing = False
                break
        
        if playing:
            choice = input("\nPlay another round? (y/n): ").lower()
            if choice != 'y':
                playing = False
