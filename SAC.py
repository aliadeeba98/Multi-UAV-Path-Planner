import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

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

STATE_SIZE = 4
ACTION_SIZE = 4

# =========================
# HYPERPARAMETERS
# =========================
gamma = 0.99
tau = 0.005
alpha = 0.2  # entropy coefficient

learning_rate = 0.0003
batch_size = 64
memory_size = 10000

ACTIONS = [0,1,2,3]
ACTION_MAP = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

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
# NETWORKS
# =========================
def build_actor():
    inputs = layers.Input(shape=(STATE_SIZE,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(ACTION_SIZE, activation='softmax')(x)
    return models.Model(inputs, outputs)

def build_critic():
    state_input = layers.Input(shape=(STATE_SIZE,))
    action_input = layers.Input(shape=(ACTION_SIZE,))

    x = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    q = layers.Dense(1)(x)

    return models.Model([state_input, action_input], q)

# =========================
# SAC AGENT
# =========================
class SACAgent:
    def __init__(self):
        self.actor = build_actor()
        self.critic1 = build_critic()
        self.critic2 = build_critic()

        self.target_critic1 = build_critic()
        self.target_critic2 = build_critic()

        self.update_targets(1.0)

        self.memory = deque(maxlen=memory_size)

        self.actor_optimizer = optimizers.Adam(learning_rate)
        self.critic1_optimizer = optimizers.Adam(learning_rate)
        self.critic2_optimizer = optimizers.Adam(learning_rate)

    def update_targets(self, tau_val=tau):
        for target, source in zip(self.target_critic1.weights, self.critic1.weights):
            target.assign(tau_val * source + (1 - tau_val) * target)

        for target, source in zip(self.target_critic2.weights, self.critic2.weights):
            target.assign(tau_val * source + (1 - tau_val) * target)

    def get_action(self, state):
        probs = self.actor.predict(state.reshape(1,-1), verbose=0)[0]
        return np.random.choice(ACTIONS, p=probs)

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train(self):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        actions_onehot = tf.one_hot(actions, ACTION_SIZE)

        # Critic update
        with tf.GradientTape(persistent=True) as tape:
            next_probs = self.actor(next_states)
            next_log_probs = tf.math.log(next_probs + 1e-10)

            q1_target = self.target_critic1([next_states, next_probs])
            q2_target = self.target_critic2([next_states, next_probs])
            q_target = tf.minimum(q1_target, q2_target)

            target = rewards + gamma * (1 - dones) * (q_target - alpha * next_log_probs)

            q1 = self.critic1([states, actions_onehot])
            q2 = self.critic2([states, actions_onehot])

            critic1_loss = tf.reduce_mean((q1 - target)**2)
            critic2_loss = tf.reduce_mean((q2 - target)**2)

        grads1 = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        grads2 = tape.gradient(critic2_loss, self.critic2.trainable_variables)

        self.critic1_optimizer.apply_gradients(zip(grads1, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(grads2, self.critic2.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            log_probs = tf.math.log(probs + 1e-10)

            q1 = self.critic1([states, probs])
            q2 = self.critic2([states, probs])
            q = tf.minimum(q1, q2)

            actor_loss = tf.reduce_mean(alpha * log_probs - q)

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        self.update_targets()

# =========================
# INITIALIZE AGENTS
# =========================
agents = [SACAgent() for _ in range(NUM_AGENTS)]

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
        actions = [agents[i].get_action(states[i]) for i in range(NUM_AGENTS)]

        next_states, collisions, done = env.step(actions)

        for i in range(NUM_AGENTS):
            reward = compute_reward(env.positions[i], env.goals[i], collisions)
            agents[i].remember(states[i], actions[i], reward, next_states[i], done)
            agents[i].train()

        states = next_states
        steps += 1

    if done:
        success_train += 1
        success_flag = 1
    else:
        success_flag = 0

    running_success_rate = (success_train / (ep + 1)) * 100

    print(f"Episode: {ep+1}, Steps: {steps}, Success: {success_flag}, Running Success Rate: {running_success_rate:.2f}%")

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
        actions = [np.argmax(agent.actor.predict(states[i].reshape(1,-1), verbose=0)[0])
                   for i, agent in enumerate(agents)]

        next_states, collisions, done = env.step(actions)

        collisions_episode += collisions
        states = next_states
        steps += 1

    if done:
        success_count += 1

    total_steps += steps
    total_collisions += collisions_episode

print("\n===== FINAL RESULTS =====\n")
print(f"Success Rate: {(success_count / EPISODES_TEST)*100:.2f}%")
print(f"Average Steps: {total_steps / EPISODES_TEST:.2f}")
print(f"Average Collisions: {total_collisions / EPISODES_TEST:.2f}")