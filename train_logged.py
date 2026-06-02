"""
train_logged.py — Same self-play DQN training as train.py, but logs metrics
to a CSV every LOG_EVERY hands so the learning curve can be plotted.

Non-destructive: checkpoints go to checkpoints_logged/ (NOT the original
checkpoints/), so your existing 1M-hand models are untouched.

The interesting learning happens early (epsilon hits its 0.05 floor by ~30k
hands), so the default 150k-hand run captures the full curve in ~10 minutes.

Usage:
    python3 train_logged.py            # 150,000 hands (default)
    python3 train_logged.py 300000     # custom hand count
"""

import os
import sys
import csv
import numpy as np
from collections import deque

from utils import PokerDealer
from PokerAI import SmartAIPlayer
from train import play_hand, reset_stacks, STARTING_STACK

LOG_EVERY = 2_000          # finer granularity than train.py for a smooth curve
CHECKPOINT_DIR = "checkpoints_logged"
CSV_PATH = "training_log.csv"


def main():
    total_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 55)
    print("  PokerAI Self-Play Training (with metric logging)")
    print(f"  Hands: {total_hands:,}  |  logging every {LOG_EVERY:,}")
    print(f"  Metrics -> {CSV_PATH}   Checkpoints -> {CHECKPOINT_DIR}/")
    print("=" * 55)

    p1 = SmartAIPlayer("AI_1", STARTING_STACK)
    p2 = SmartAIPlayer("AI_2", STARTING_STACK)
    players = [p1, p2]
    dealer = PokerDealer([p.name for p in players])
    button = 0

    window = LOG_EVERY
    win_hist = {p.name: deque(maxlen=window) for p in players}
    rew_hist = {p.name: deque(maxlen=window) for p in players}
    loss_hist = {p.name: deque(maxlen=window) for p in players}

    rows = []

    for hand_num in range(1, total_hands + 1):
        if p1.stack <= 0 or p2.stack <= 0:
            reset_stacks(players, STARTING_STACK)

        deltas, winners = play_hand(players, dealer, button)

        for p in players:
            reward = float(deltas[p.name])
            p.store_hand_transitions(reward)
            loss = p.learn()
            rew_hist[p.name].append(reward)
            win_hist[p.name].append(1 if p.name in winners else 0)
            if loss is not None:
                loss_hist[p.name].append(loss)

        button = (button + 1) % 2

        if hand_num % LOG_EVERY == 0:
            row = {
                "hand": hand_num,
                "epsilon": round(p1.epsilon, 5),
                "ai1_win_rate": round(np.mean(win_hist[p1.name]) * 100, 2) if win_hist[p1.name] else 0,
                "ai2_win_rate": round(np.mean(win_hist[p2.name]) * 100, 2) if win_hist[p2.name] else 0,
                "ai1_avg_reward": round(np.mean(rew_hist[p1.name]), 2) if rew_hist[p1.name] else 0,
                "ai2_avg_reward": round(np.mean(rew_hist[p2.name]), 2) if rew_hist[p2.name] else 0,
                "ai1_loss": round(np.mean(loss_hist[p1.name]), 5) if loss_hist[p1.name] else float("nan"),
                "ai2_loss": round(np.mean(loss_hist[p2.name]), 5) if loss_hist[p2.name] else float("nan"),
            }
            rows.append(row)
            print(
                f"Hand {hand_num:>7,} | eps {row['epsilon']:.3f} | "
                f"WR {row['ai1_win_rate']:5.1f}/{row['ai2_win_rate']:5.1f} | "
                f"reward {row['ai1_avg_reward']:+7.1f}/{row['ai2_avg_reward']:+7.1f} | "
                f"loss {row['ai1_loss']:.3f}/{row['ai2_loss']:.3f}"
            )

    # Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved metrics to {CSV_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
