import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from utils import evaluate_hand_counts, PokerPlayer
from PokerModel import PokerEVNet
from PokerMLUtils import vectorize_state
from ReplayBuffer import ReplayBuffer

GAMMA = 0.99
BATCH_SIZE = 512
MIN_BUFFER_SIZE = 512
LEARNING_RATE = 1e-5

class SmartAIPlayer(PokerPlayer):
    def __init__(self, name, stack):
        self.memory = []          # Temporary storage for current hand's transitions
        self.epsilon = 1.0        # Start with 100% random moves to explore
        self.epsilon_min = 0.05   # Minimum exploration
        self.epsilon_decay = 0.9999
        super().__init__(name, stack)

        # Brain
        self.model = PokerEVNet(input_size=109, output_size=5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=50000)

    def decide_action(self, highest_bet, to_call, community_cards, pot_size, opponent, is_dealer):
        state = vectorize_state(self, opponent, community_cards, pot_size, highest_bet, is_dealer)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action_idx = random.randint(0, 4)
        else:
            self.model.eval()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values).item()

        # Save state/action to hand memory (reward & next_state filled in after hand)
        self.memory.append({
            'state': state,
            'action': action_idx
        })

        # Translate action index to game engine string
        if action_idx == 0:
            return "fold" if to_call > 0 else "check"

        if action_idx == 1:
            return "call" if to_call > 0 else "check"

        if action_idx == 4:
            return f"raise {self.stack + self.current_bet}"

        pot_multipliers = {2: 0.33, 3: 0.75}
        multiplier = pot_multipliers[action_idx]
        bet_increment = int(pot_size * multiplier)
        target_total = max(highest_bet + 20, highest_bet + bet_increment)
        return f"raise {target_total}"

    def store_hand_transitions(self, reward):
        """
        Called after a hand ends. Assigns the final hand reward to every
        transition in self.memory and pushes them all to the replay buffer.
        """
        for i in range(len(self.memory)):
            state = self.memory[i]['state']
            action = self.memory[i]['action']

            if i + 1 < len(self.memory):
                next_state = self.memory[i + 1]['state']
                done = 0
            else:
                next_state = np.zeros(109, dtype=np.float32)
                done = 1

            self.replay_buffer.push(state, action, reward, next_state, done)

        # Clear hand memory and decay epsilon
        self.memory = []
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self):
        """
        One DQN update step. Samples a batch from the replay buffer,
        computes Bellman targets, and backprops.
        Returns the scalar loss value, or None if buffer isn't ready.
        """
        if len(self.replay_buffer) < MIN_BUFFER_SIZE:
            return None

        self.model.train()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions)
        rewards_t     = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones)

        # Current Q-values for the actions that were actually taken
        current_q = self.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Bellman target: r + gamma * max_a Q(s', a)  [zero out terminal states]
        with torch.no_grad():
            next_q = self.model(next_states_t).max(1)[0]
            target_q = rewards_t + GAMMA * next_q * (1 - dones_t)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
