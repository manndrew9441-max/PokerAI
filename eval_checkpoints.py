"""
eval_checkpoints.py — Reconstruct the PokerAI learning curve from saved checkpoints.

Training (train.py) saved a model checkpoint every 100k hands but didn't log
metrics. This script rebuilds the learning curve by *evaluating* each checkpoint:

  For each checkpoint (100k, 200k, ... 1M hands of training):
    - Load the model into a greedy player (epsilon = 0, no exploration)
    - Play N hands of heads-up Hold'em against a random baseline
    - Record chips won per 100 hands (bb/100) and hand win rate

If training worked, both metrics rise across checkpoints. The result is saved to
checkpoint_eval.json and plotted to learning_curve.png.

Usage:
    python3 eval_checkpoints.py                # default 10,000 hands/checkpoint
    python3 eval_checkpoints.py 20000          # custom hands/checkpoint
    python3 eval_checkpoints.py 20000 AI_2     # evaluate AI_2's checkpoints
"""

import os
import re
import sys
import json
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PokerAI import SmartAIPlayer
from utils import PokerDealer
from train import play_hand, reset_stacks, STARTING_STACK, BB_AMOUNT

CHECKPOINT_DIR = "checkpoints"


def load_greedy_player(name, ckpt_path):
    """Load a checkpoint into a player that always acts greedily (epsilon=0)."""
    p = SmartAIPlayer(name, STARTING_STACK)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    p.model.load_state_dict(ckpt["model_state"])
    p.model.eval()
    p.epsilon = 0.0  # pure exploitation — no random moves
    return p


def make_random_player(name):
    """A baseline that always acts randomly (epsilon=1 → never uses the model)."""
    p = SmartAIPlayer(name, STARTING_STACK)
    p.epsilon = 1.0
    return p


def evaluate(ckpt_path, n_hands):
    """
    Play n_hands of heads-up between the checkpoint (greedy) and a random
    baseline. Button alternates each hand for positional fairness. Stacks reset
    on bust (same as training). Returns (chips_per_100, win_rate_pct, net_chips).
    """
    agent = load_greedy_player("AGENT", ckpt_path)
    rng_player = make_random_player("RANDOM")
    players = [agent, rng_player]
    dealer = PokerDealer([p.name for p in players])

    net_chips = 0.0
    agent_wins = 0
    button = 0

    for _ in range(n_hands):
        if agent.stack <= 0 or rng_player.stack <= 0:
            reset_stacks(players, STARTING_STACK)

        deltas, winners = play_hand(players, dealer, button)
        net_chips += deltas["AGENT"]
        if "AGENT" in winners:
            agent_wins += 1

        # Clear per-hand action memory so it doesn't grow unbounded (we never learn here)
        agent.memory = []
        rng_player.memory = []
        button = (button + 1) % 2

    chips_per_100 = (net_chips / n_hands) * 100
    bb_per_100 = chips_per_100 / BB_AMOUNT
    win_rate = (agent_wins / n_hands) * 100
    return chips_per_100, bb_per_100, win_rate, net_chips


def main():
    n_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    tag = sys.argv[2] if len(sys.argv) > 2 else "AI_1"

    # Collect checkpoints like AI_1_hand_100000.pth, sorted by hand count.
    pattern = os.path.join(CHECKPOINT_DIR, f"{tag}_hand_*.pth")
    paths = glob.glob(pattern)
    def hand_num(p):
        m = re.search(r"_hand_(\d+)\.pth$", p)
        return int(m.group(1)) if m else 0
    paths = sorted(paths, key=hand_num)

    if not paths:
        print(f"No checkpoints matching {pattern}")
        sys.exit(1)

    print(f"Evaluating {len(paths)} checkpoints of {tag}, {n_hands:,} hands each, vs random baseline\n")
    print(f"{'Hands trained':>14} {'chips/100':>12} {'bb/100':>9} {'win rate':>10}")
    print("-" * 50)

    results = []
    for path in paths:
        h = hand_num(path)
        cph, bb100, wr, net = evaluate(path, n_hands)
        results.append({
            "hands_trained": h,
            "chips_per_100": round(cph, 2),
            "bb_per_100": round(bb100, 2),
            "win_rate": round(wr, 2),
            "net_chips": round(net, 2),
        })
        print(f"{h:>14,} {cph:>+12.1f} {bb100:>+9.2f} {wr:>9.1f}%")

    # Save raw results
    out_json = "checkpoint_eval.json"
    with open(out_json, "w") as f:
        json.dump({"tag": tag, "hands_per_eval": n_hands, "results": results}, f, indent=2)
    print(f"\nSaved {out_json}")

    # ── Plot ──
    xs = [r["hands_trained"] / 1000 for r in results]   # in thousands
    bb = [r["bb_per_100"] for r in results]
    wr = [r["win_rate"] for r in results]

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"PokerAI Learning Curve — {tag} vs Random Baseline\n"
        f"(Deep Q-Network, {n_hands:,} hands evaluated per checkpoint)",
        fontsize=13, fontweight="bold",
    )

    ax1.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax1.plot(xs, bb, marker="o", color="#38a169", linewidth=2, markersize=6)
    ax1.set_ylabel("Big blinds won / 100 hands")
    ax1.set_title("Profitability vs training (higher = stronger play)", fontsize=10, color="#aaa")
    ax1.grid(True, alpha=0.2)

    ax2.axhline(50, color="#888", linewidth=0.8, linestyle="--", label="50% (coin flip)")
    ax2.plot(xs, wr, marker="o", color="#3182ce", linewidth=2, markersize=6)
    ax2.set_ylabel("Hand win rate (%)")
    ax2.set_xlabel("Hands trained (thousands)")
    ax2.set_title("Win rate vs training", fontsize=10, color="#aaa")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="best", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = "learning_curve.png"
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
