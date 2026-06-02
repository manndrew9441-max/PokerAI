# PokerAI — Deep Q-Network for Heads-Up Texas Hold'em

A poker-playing AI trained from scratch with deep reinforcement learning. Two
neural-network agents learn by playing each other over a million hands of
heads-up No-Limit Texas Hold'em — no hand-coded strategy, no libraries doing the
RL for it. Just a Q-network, a replay buffer, and self-play.

## What it does

- Learns a poker strategy purely from self-play and reward signals (chips won/lost)
- Reaches a policy that beats a random baseline by **+150 to +500 big blinds per 100 hands**
- Plays a human opponent through a terminal interface using the trained model

## How it works

**State → action values (Q-learning).** Every decision point is encoded as a
109-feature vector and fed to a neural network that outputs an estimated value
for each of 5 actions (fold, call/check, small raise, big raise, all-in).

State vector (`PokerMLUtils.vectorize_state`, 109 features):
- 52 — one-hot encoding of hole cards
- 52 — one-hot encoding of community cards
- 5 — metadata: stack ratio, pot odds, position, betting street, aggression level

Network (`PokerModel.PokerEVNet`):
```
109 → 256 → 128 → 64 → 5     (ReLU activations, outputs = Q-values per action)
```

Learning (`PokerAI.SmartAIPlayer.learn`), standard DQN:
- **Experience replay** — a 50k-transition buffer; minibatches are sampled
  randomly to decorrelate sequential hands
- **Bellman target** — `Q(s,a) ← r + γ · maxₐ Q(s',a)`, γ = 0.99
- **Epsilon-greedy exploration** — starts 100% random, decays to a 5% floor as
  the agent gains confidence
- **Gradient clipping** — stabilizes updates against large reward swings

**Self-play.** Two agents (`AI_1`, `AI_2`) train simultaneously against each
other (`train.py`). Because they improve in lockstep, their head-to-head win rate
stays near 50% by design — learning shows up as falling loss, not one agent
beating the other.

## Results

| Plot | Shows |
|---|---|
| `training_curve.png` | The learning process — Q-value loss falls ~30× and exploration anneals from 100% to 5% over 150k hands |
| `learning_curve.png` | The outcome — trained checkpoints evaluated vs a random baseline; the agent is strongly profitable at every stage |

## Layout

| File | Purpose |
|---|---|
| `PokerModel.py` | The Q-network (`PokerEVNet`) |
| `PokerAI.py` | The RL agent — action selection, replay buffer, DQN update |
| `PokerMLUtils.py` | State vectorization (the 109-feature encoding) |
| `ReplayBuffer.py` | Experience replay buffer |
| `utils.py` | Game engine — dealer, hand evaluation, betting rules |
| `train.py` | Self-play training loop (1M hands, checkpoints every 100k) |
| `train_logged.py` | Same training with per-interval metric logging to CSV |
| `eval_checkpoints.py` | Evaluate saved checkpoints vs a random baseline |
| `plot_training.py` | Plot the training curve from the metrics CSV |
| `poker.py` | Play the trained AI yourself in the terminal |

## Reproduce

```bash
pip install torch numpy matplotlib

# Train with metric logging (~6 min for 150k hands; checkpoints -> checkpoints_logged/)
python3 train_logged.py 150000
python3 plot_training.py              # -> training_curve.png

# Evaluate trained checkpoints vs random and plot strength over training
python3 eval_checkpoints.py 10000     # -> learning_curve.png, checkpoint_eval.json

# Play the trained AI yourself
python3 poker.py                      # loads checkpoints/AI_1_final_hand_1000000.pth
```

## Notes

- Rewards are raw chip deltas, so the loss magnitude is large in absolute terms;
  the meaningful signal is the relative ~30× decline as the policy converges.
- Hand strength is evaluated with a full 7-card evaluator (`evaluate_hand_counts`)
  covering all hand ranks including straight/flush/straight-flush edge cases.
