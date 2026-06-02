"""
train.py — Self-play DQN training loop for PokerAI
Two SmartAIPlayer instances play each other. Both learn simultaneously.
Checkpoints saved every 1,000 hands to checkpoints/.
"""

import os
import random
import torch
import numpy as np
from collections import deque

from utils import PokerDealer, format_cards
from PokerAI import SmartAIPlayer

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TOTAL_HANDS      = 1_000_000
STARTING_STACK   = 1_000
SB_AMOUNT        = 10
BB_AMOUNT        = 20
CHECKPOINT_EVERY = 100_000
LOG_EVERY        = 10_000          # Print rolling stats every N hands
CHECKPOINT_DIR   = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def save_checkpoint(player, hand_num, tag):
    path = os.path.join(CHECKPOINT_DIR, f"{tag}_hand_{hand_num}.pth")
    torch.save({
        'hand': hand_num,
        'model_state': player.model.state_dict(),
        'optimizer_state': player.optimizer.state_dict(),
        'epsilon': player.epsilon,
    }, path)
    print(f"  [CKPT] Saved {path}")


def reset_stacks(players, starting_stack):
    for p in players:
        p.stack = starting_stack


def run_betting_phase(players, dealer_engine, phase_name,
                      button_index, initial_highest=0, start_idx=0):
    """
    Stripped-down betting phase (no human I/O).
    Returns immediately if only one player remains active.
    """
    highest_bet = initial_highest
    players_acted = {p.name: False for p in players}

    if phase_name != "Pre-Flop":
        for p in players:
            p.reset_for_new_round()
        highest_bet = 0

    while True:
        active = [p for p in players if not p.is_folded and not p.is_all_in]

        if len(active) <= 1:
            break

        if (all(p.current_bet == highest_bet for p in active) and
                all(players_acted[p.name] for p in active)):
            break

        acted_this_pass = False
        for i in range(len(players)):
            curr_idx = (start_idx + i) % len(players)
            p = players[curr_idx]

            if p.is_folded or p.is_all_in:
                continue
            if p.current_bet == highest_bet and players_acted[p.name]:
                continue

            to_call = highest_bet - p.current_bet
            opponent = players[0] if p == players[1] else players[1]
            is_dealer = (button_index == players.index(p))

            action = p.decide_action(
                highest_bet, to_call,
                dealer_engine.community_cards,
                dealer_engine.pot,
                opponent, is_dealer
            )
            players_acted[p.name] = True
            acted_this_pass = True

            if action == "fold":
                p.fold()

            elif action in ["call", "check"]:
                dealer_engine.collect_bets(p.place_bet(to_call))

            elif action.startswith("raise"):
                try:
                    new_total = int(action.split()[1])
                    if new_total <= highest_bet:
                        dealer_engine.collect_bets(p.place_bet(to_call))
                    else:
                        diff = new_total - p.current_bet
                        highest_bet = new_total
                        dealer_engine.collect_bets(p.place_bet(diff))
                        for name in players_acted:
                            if name != p.name:
                                players_acted[name] = False
                except (ValueError, IndexError):
                    dealer_engine.collect_bets(p.place_bet(to_call))

        if not acted_this_pass:
            break


def evaluate_showdown(players, dealer_engine):
    """
    Returns the winner(s) and awards the pot.
    Returns list of winner names.
    """
    from utils import evaluate_hand_counts

    active = [p for p in players if not p.is_folded]

    if len(active) == 1:
        active[0].stack += dealer_engine.pot
        dealer_engine.pot = 0
        return [active[0].name]

    results = [(p, evaluate_hand_counts(p.hand, dealer_engine.community_cards))
               for p in active]
    results.sort(key=lambda x: x[1], reverse=True)
    best_score = results[0][1]
    winners = [p for p, score in results if score == best_score]

    split = dealer_engine.pot // len(winners)
    for w in winners:
        w.stack += split
    winners[0].stack += dealer_engine.pot % len(winners)  # dust chips

    dealer_engine.pot = 0
    return [w.name for w in winners]


def play_hand(players, dealer_engine, button_index):
    """
    Plays one complete hand of heads-up Texas Hold'em.
    Returns dict of stack changes per player name.
    """
    initial_stacks = {p.name: p.stack for p in players}

    # Deal
    dealer_engine.suffle_and_deal()
    sb_idx = button_index
    bb_idx = (button_index + 1) % 2
    sb_p = players[sb_idx]
    bb_p = players[bb_idx]

    for p in players:
        p.recieve_cards(dealer_engine.player[p.name])
        p.reset_for_new_round()
        p.is_all_in = False   # Full reset between hands
        p.is_folded = False

    # Blinds
    dealer_engine.collect_bets(sb_p.place_bet(SB_AMOUNT))
    dealer_engine.collect_bets(bb_p.place_bet(BB_AMOUNT))

    # Pre-Flop (dealer/SB acts first in heads-up)
    run_betting_phase(players, dealer_engine, "Pre-Flop",
                      button_index, initial_highest=BB_AMOUNT, start_idx=sb_idx)

    # Flop
    if len([p for p in players if not p.is_folded]) > 1:
        dealer_engine.deal_flop()
        run_betting_phase(players, dealer_engine, "Flop",
                          button_index, start_idx=bb_idx)

    # Turn
    if len([p for p in players if not p.is_folded]) > 1:
        dealer_engine.deal_next_card()
        run_betting_phase(players, dealer_engine, "Turn",
                          button_index, start_idx=bb_idx)

    # River
    if len([p for p in players if not p.is_folded]) > 1:
        dealer_engine.deal_next_card()
        run_betting_phase(players, dealer_engine, "River",
                          button_index, start_idx=bb_idx)

    # Showdown
    winners = evaluate_showdown(players, dealer_engine)

    # Stack deltas
    deltas = {p.name: p.stack - initial_stacks[p.name] for p in players}
    return deltas, winners


# ─────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train():
    print("=" * 55)
    print("  PokerAI Self-Play Training")
    print(f"  Hands: {TOTAL_HANDS:,}  |  Stacks: {STARTING_STACK}")
    print(f"  Checkpoints every {CHECKPOINT_EVERY:,} hands → {CHECKPOINT_DIR}/")
    print("=" * 55)

    p1 = SmartAIPlayer("AI_1", STARTING_STACK)
    p2 = SmartAIPlayer("AI_2", STARTING_STACK)
    players = [p1, p2]

    dealer_engine = PokerDealer([p.name for p in players])
    button_index = 0

    # Rolling stats (last 1,000 hands)
    window = 1000
    win_history   = {p.name: deque(maxlen=window) for p in players}
    reward_history = {p.name: deque(maxlen=window) for p in players}
    loss_history   = {p.name: deque(maxlen=window) for p in players}

    total_wins = {p.name: 0 for p in players}

    for hand_num in range(1, TOTAL_HANDS + 1):

        # Re-buy if either player busts (keeps training going)
        if p1.stack <= 0 or p2.stack <= 0:
            reset_stacks(players, STARTING_STACK)

        # ── Play one hand ──
        deltas, winners = play_hand(players, dealer_engine, button_index)

        # ── Store transitions & learn ──
        for p in players:
            reward = float(deltas[p.name])
            p.store_hand_transitions(reward)

            loss = p.learn()

            reward_history[p.name].append(reward)
            win_history[p.name].append(1 if p.name in winners else 0)
            if loss is not None:
                loss_history[p.name].append(loss)

        for w in winners:
            total_wins[w] += 1

        # Rotate dealer button
        button_index = (button_index + 1) % 2

        # ── Logging ──
        if hand_num % LOG_EVERY == 0:
            p1_wr  = np.mean(win_history[p1.name]) * 100 if win_history[p1.name] else 0
            p2_wr  = np.mean(win_history[p2.name]) * 100 if win_history[p2.name] else 0
            p1_avg = np.mean(reward_history[p1.name]) if reward_history[p1.name] else 0
            p2_avg = np.mean(reward_history[p2.name]) if reward_history[p2.name] else 0
            p1_loss = np.mean(loss_history[p1.name]) if loss_history[p1.name] else float('nan')
            p2_loss = np.mean(loss_history[p2.name]) if loss_history[p2.name] else float('nan')

            print(
                f"Hand {hand_num:>7,} | "
                f"WR: {p1.name} {p1_wr:5.1f}% / {p2.name} {p2_wr:5.1f}% | "
                f"Avg Reward: {p1_avg:+7.1f} / {p2_avg:+7.1f} | "
                f"Loss: {p1_loss:.4f} / {p2_loss:.4f} | "
                f"ε: {p1.epsilon:.3f}"
            )

        # ── Checkpoints ──
        if hand_num % CHECKPOINT_EVERY == 0:
            save_checkpoint(p1, hand_num, p1.name)
            save_checkpoint(p2, hand_num, p2.name)

    # ── Final summary ──
    print("\n" + "=" * 55)
    print("  Training Complete")
    print(f"  {p1.name} total wins: {total_wins[p1.name]:,}")
    print(f"  {p2.name} total wins: {total_wins[p2.name]:,}")
    print("=" * 55)
    save_checkpoint(p1, TOTAL_HANDS, f"{p1.name}_final")
    save_checkpoint(p2, TOTAL_HANDS, f"{p2.name}_final")


if __name__ == "__main__":
    train()
