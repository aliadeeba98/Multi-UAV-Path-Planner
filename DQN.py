import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

import argparse
# =========================
# ARGUMENT PARSER
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--grid_size", type=int, default=10)
parser.add_argument("--num_agents", type=int, default=3)
parser.add_argument("--num_obstacles", type=int, default=20)

EPISODES_TRAIN = 5000
EPISODES_TEST = 1000
MAX_STEPS = 200

STATE_SIZE = 4   # (x, y, goal_x, goal_y)
ACTION_SIZE = 4

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_MAP = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

# =========================
# HYPERPARAMETERS
# =========================
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
learning_rate = 0.001
batch_size = 64
memory_size = 5000

# =========================
# ENVIRONMENT
# =========================
class GridWorld:
    def __init__(self):
        self.obstacles = self.generate_obstacles()
        self.starts = self.random_positions()
        self.goals = self.random_positions()

    def generate_obstacles(self):
        obstacles = set()
        while len(obstacles) < NUM_OBSTACLES:
            obstacles.add((random.randint(0, GRID_SIZE-1),
                           random.randint(0, GRID_SIZE-1)))
        return obstacles

    def random_positions(self):
        positions = []
        while len(positions) < NUM_AGENTS:
            pos = (random.randint(0, GRID_SIZE-1),
                   random.randint(0, GRID_SIZE-1))
            if pos not in self.obstacles and pos not in positions:
                positions.append(pos)
        return positions

    def reset(self):
        self.positions = list(self.starts)
        return self.get_states()

    def get_states(self):
        states = []
        for i in range(NUM_AGENTS):
            x, y = self.positions[i]
            gx, gy = self.goals[i]
            states.append(np.array([x, y, gx, gy]) / GRID_SIZE)
        return states

    def step(self, actions):
        new_positions = []
        collisions = 0

        for i, action in enumerate(actions):
            move = ACTION_MAP[action]
            new_pos = (self.positions[i][0] + move[0],
                       self.positions[i][1] + move[1])

            if not (0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE):
                new_pos = self.positions[i]

            if new_pos in self.obstacles:
                new_pos = self.positions[i]
                collisions += 1

            new_positions.append(new_pos)

        if len(set(new_positions)) < len(new_positions):
            collisions += 1

        self.positions = new_positions

        done = all(self.positions[i] == self.goals[i] for i in range(NUM_AGENTS))

        return self.get_states(), collisions, done

# =========================
# DQN MODEL
# =========================
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=STATE_SIZE, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(ACTION_SIZE, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate))
    return model

# =========================
# AGENT CLASS
# =========================
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=memory_size)
        self.model = build_model()
        self.target_model = build_model()
        self.update_target()

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() < epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def replay(self):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for s, a, r, s_next, done in batch:
            target = r
            if not done:
                target += gamma * np.max(self.target_model.predict(s_next.reshape(1, -1), verbose=0)[0])

            target_f = self.model.predict(s.reshape(1, -1), verbose=0)
            target_f[0][a] = target

            self.model.fit(s.reshape(1, -1), target_f, epochs=1, verbose=0)

# =========================
# INITIALIZE AGENTS
# =========================
agents = [DQNAgent() for _ in range(NUM_AGENTS)]

# =========================
# REWARD FUNCTION
# =========================
def compute_reward(pos, goal, collision):
    if pos == goal:
        return 100
    if collision:
        return -10
    return -1

# =========================
# TRAINING
# =========================
print("\n===== TRAINING STARTED =====\n")

success_train = 0

for ep in range(EPISODES_TRAIN):
    env = GridWorld()
    states = env.reset()

    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        actions = []

        for i in range(NUM_AGENTS):
            action = agents[i].act(states[i])
            actions.append(action)

        next_states, collisions, done = env.step(actions)

        for i in range(NUM_AGENTS):
            reward = compute_reward(env.positions[i], env.goals[i], collisions)

            agents[i].remember(states[i], actions[i], reward, next_states[i], done)
            agents[i].replay()

        states = next_states
        steps += 1

    if done:
        success_train += 1
        success_flag = 1
    else:
        success_flag = 0

    running_success_rate = (success_train / (ep + 1)) * 100

    print(f"Episode: {ep+1}, Steps: {steps}, Success: {success_flag}, Running Success Rate: {running_success_rate:.2f}%")

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Update target networks periodically
    if ep % 50 == 0:
        for agent in agents:
            agent.update_target()

print("\n===== TRAINING COMPLETED =====\n")

# =========================
# TESTING
# =========================
print("\n===== TESTING STARTED =====\n")

success_count = 0
total_steps = 0
total_collisions = 0

for ep in range(EPISODES_TEST):
    env = GridWorld()
    states = env.reset()

    done = False
    steps = 0
    collisions_episode = 0

    while not done and steps < MAX_STEPS:
        actions = []

        for i in range(NUM_AGENTS):
            q_values = agents[i].model.predict(states[i].reshape(1, -1), verbose=0)
            actions.append(np.argmax(q_values[0]))

        next_states, collisions, done = env.step(actions)

        collisions_episode += collisions
        states = next_states
        steps += 1

    if done:
        success_count += 1

    total_steps += steps
    total_collisions += collisions_episode

# =========================
# RESULTS
# =========================
success_rate = (success_count / EPISODES_TEST) * 100
avg_steps = total_steps / EPISODES_TEST
avg_collisions = total_collisions / EPISODES_TEST

print("\n===== FINAL RESULTS =====\n")
print(f"Grid Size: {GRID_SIZE}")
print(f"Agents: {NUM_AGENTS}")
print(f"Number of Obstacles: {NUM_OBSTACLES}")

print(f"Success Rate: {success_rate:.2f}%")
print(f"Average Steps: {avg_steps:.2f}")
print(f"Average Collisions: {avg_collisions:.2f}")
