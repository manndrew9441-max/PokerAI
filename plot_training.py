"""
plot_training.py — Plot the DQN learning curve from training_log.csv.

Produces training_curve.png with two panels:
  1. Training loss (log scale) for both self-play agents — the core learning
     signal. Falls ~30x as the network learns to predict action values.
  2. Epsilon decay — exploration annealing from 100% random down to the 5% floor.

Self-play note: both agents train against each other, so their head-to-head win
rate stays near 50% by design. Improvement shows up as falling loss, not as one
agent beating the other. (See learning_curve.png for strength vs a random
baseline.)
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "training_log.csv"
OUT_PNG = "training_curve.png"


def main():
    hands, eps = [], []
    ai1_loss, ai2_loss = [], []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            hands.append(int(row["hand"]) / 1000)  # thousands
            eps.append(float(row["epsilon"]))
            ai1_loss.append(float(row["ai1_loss"]))
            ai2_loss.append(float(row["ai2_loss"]))

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "PokerAI — Deep Q-Network Training Curve\n"
        "Self-play, 150,000 hands of heads-up Texas Hold'em",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: loss (log scale)
    ax1.plot(hands, ai1_loss, color="#38a169", linewidth=2, label="Agent 1")
    ax1.plot(hands, ai2_loss, color="#3182ce", linewidth=2, label="Agent 2")
    ax1.set_yscale("log")
    ax1.set_ylabel("Q-value loss (MSE, log scale)")
    ax1.set_title("Training loss falls ~30x as the network learns to value actions",
                  fontsize=10, color="#aaa")
    ax1.grid(True, alpha=0.2, which="both")
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: epsilon decay
    ax2.plot(hands, eps, color="#d69e2e", linewidth=2)
    ax2.axhline(0.05, color="#888", linewidth=0.8, linestyle="--", label="exploration floor (5%)")
    ax2.set_ylabel("Epsilon (exploration rate)")
    ax2.set_xlabel("Hands trained (thousands)")
    ax2.set_title("Exploration anneals from 100% random to 5% as confidence grows",
                  fontsize=10, color="#aaa")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PNG, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
