"""
DQN and SAC implementations using TensorFlow/Keras.
Exact algorithm match with ground-truth DQN.py and SAC.py.

Network architectures, reward functions, environment step logic,
and training procedures are all identical to the ground truth.
Episode counts are reduced (fixed layout converges faster than random).
"""
import numpy as np
import random
import time
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import models, layers, optimizers


# =========================
# CONSTANTS (from DQN.py / SAC.py)
# =========================
STATE_SIZE = 4   # (x, y, goal_x, goal_y) normalized
ACTION_SIZE = 4

# Action map identical to DQN.py and SAC.py
ACTION_MAP = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}

ACTIONS = [0, 1, 2, 3]


# =========================
# ENVIRONMENT
# Exact match with DQN.py / SAC.py GridWorld
# =========================
class DeepRLGridWorld:
    def __init__(self, grid_size, obstacles, starts, goals):
        self.grid_size = grid_size
        self.obstacles = set(tuple(o) for o in obstacles)
        self.starts = [tuple(s) for s in starts]
        self.goals = [tuple(g) for g in goals]
        self.num_agents = len(starts)
        self.reset()

    def reset(self):
        self.positions = list(self.starts)
        return self.get_states()

    def get_states(self):
        """State: [x, y, gx, gy] / grid_size — from DQN.py get_states()"""
        states = []
        for i in range(self.num_agents):
            x, y = self.positions[i]
            gx, gy = self.goals[i]
            states.append(np.array([x, y, gx, gy], dtype=np.float32) / self.grid_size)
        return states

    def step(self, actions):
        """Exact step logic from DQN.py / SAC.py GridWorld.step()"""
        new_positions = []
        collisions = 0

        for i, action in enumerate(actions):
            move = ACTION_MAP[action]
            new_pos = (self.positions[i][0] + move[0],
                       self.positions[i][1] + move[1])

            # Boundary check
            if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
                new_pos = self.positions[i]

            # Obstacle collision — revert position
            if new_pos in self.obstacles:
                new_pos = self.positions[i]
                collisions += 1

            new_positions.append(new_pos)

        # Inter-agent collision
        if len(set(new_positions)) < len(new_positions):
            collisions += 1

        self.positions = new_positions
        done = all(self.positions[i] == self.goals[i] for i in range(self.num_agents))

        return self.get_states(), collisions, done


# =========================
# REWARD FUNCTION
# Exact match with DQN.py / SAC.py compute_reward()
# =========================
def compute_reward(pos, goal, collision):
    if pos == goal:
        return 100
    if collision:
        return -10
    return -1


# =========================
# PATH TRIMMING
# Stop each UAV's path at the first time it reaches its goal
# =========================
def trim_paths_at_goals(paths, goals):
    trimmed = []
    for i, path in enumerate(paths):
        goal = goals[i]
        trimmed_path = []
        for pos in path:
            trimmed_path.append(pos)
            if pos[0] == goal[0] and pos[1] == goal[1]:
                break
        trimmed.append(trimmed_path)
    return trimmed


# ================================================================
#  DQN — Exact algorithm from DQN.py
# ================================================================

def build_dqn_model(lr):
    """
    Exact architecture from DQN.py build_model():
    Dense(64, relu) → Dense(64, relu) → Dense(4, linear)
    Compiled with MSE loss + Adam optimizer.
    """
    model = models.Sequential([
        layers.Input(shape=(STATE_SIZE,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(ACTION_SIZE, activation='linear')
    ])
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=lr))
    return model


class DQNAgent:
    """Exact match with DQN.py DQNAgent class."""

    def __init__(self, lr=0.001, mem_size=5000):
        self.memory = deque(maxlen=mem_size)
        self.model = build_dqn_model(lr)
        self.target_model = build_dqn_model(lr)
        self.update_target()

    def update_target(self):
        """From DQN.py: copy main model weights to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, epsilon):
        """
        From DQN.py DQNAgent.act():
        ε-greedy action selection using Q-value predictions.
        """
        if np.random.rand() < epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model(state.reshape(1, -1), training=False).numpy()
        return int(np.argmax(q_values[0]))

    def remember(self, s, a, r, s_next, done):
        """From DQN.py: store transition in replay memory."""
        self.memory.append((s, a, r, s_next, done))

    def replay(self, batch_size, gamma):
        """
        From DQN.py DQNAgent.replay():
        Sample batch from memory → compute TD targets → update model.
        Vectorized for speed (single batch predict + single train_on_batch).
        """
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        states_b = np.array([b[0] for b in batch])
        actions_b = np.array([b[1] for b in batch])
        rewards_b = np.array([b[2] for b in batch], dtype=np.float32)
        next_states_b = np.array([b[3] for b in batch])
        dones_b = np.array([b[4] for b in batch], dtype=np.float32)

        # Batch target computation (mathematically identical to per-sample in DQN.py)
        target_q_next = self.target_model(next_states_b, training=False).numpy()
        target_f = self.model(states_b, training=False).numpy()

        max_q_next = np.max(target_q_next, axis=1)

        for i in range(batch_size):
            target = rewards_b[i]
            if not dones_b[i]:
                target += gamma * max_q_next[i]
            target_f[i][actions_b[i]] = target

        self.model.train_on_batch(states_b, target_f)


def run_dqn(grid_size, obstacles, starts, goals):
    """
    Full DQN training + path generation.
    Exact algorithm structure from DQN.py, reduced episodes (fixed layout).
    """
    # Hyperparameters from DQN.py
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    learning_rate = 0.001
    batch_size = 64
    memory_size = 5000

    # Exact match with DQN.py ground truth
    EPISODES_TRAIN = 5000
    MAX_STEPS = 200

    num_agents = len(starts)
    env = DeepRLGridWorld(grid_size, obstacles, starts, goals)
    agents = [DQNAgent(lr=learning_rate, mem_size=memory_size) for _ in range(num_agents)]

    # === TRAINING (exact structure from DQN.py training loop) ===
    for ep in range(EPISODES_TRAIN):
        states = env.reset()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            actions = []
            for i in range(num_agents):
                action = agents[i].act(states[i], epsilon)
                actions.append(action)

            next_states, collisions, done = env.step(actions)

            for i in range(num_agents):
                reward = compute_reward(env.positions[i], env.goals[i], collisions)
                agents[i].remember(states[i], actions[i], reward, next_states[i], done)
                agents[i].replay(batch_size, gamma)

            states = next_states
            steps += 1

        # Epsilon decay (from DQN.py)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Update target networks periodically (from DQN.py: every 50 episodes)
        if ep % 50 == 0:
            for agent in agents:
                agent.update_target()

    # === TESTING — generate path (exact structure from DQN.py testing loop) ===
    states = env.reset()
    paths = [[list(s)] for s in starts]
    done = False
    steps = 0
    collisions_count = 0

    while not done and steps < MAX_STEPS:
        actions = []
        for i in range(num_agents):
            # From DQN.py testing: deterministic argmax of Q-values
            q_values = agents[i].model(states[i].reshape(1, -1), training=False).numpy()
            actions.append(int(np.argmax(q_values[0])))

        next_states, collisions, done = env.step(actions)
        collisions_count += collisions

        for i in range(num_agents):
            paths[i].append(list(env.positions[i]))

        states = next_states
        steps += 1

    # Trim paths: stop each UAV at its goal
    goals_list = [list(g) for g in env.goals]
    paths = trim_paths_at_goals(paths, goals_list)

    return paths, done, steps, collisions_count


# ================================================================
#  SAC — Exact algorithm from SAC.py
# ================================================================

def build_sac_actor():
    """
    Exact architecture from SAC.py build_actor():
    Input(4) → Dense(128, relu) → Dense(128, relu) → Dense(4, softmax)
    """
    inputs = layers.Input(shape=(STATE_SIZE,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(ACTION_SIZE, activation='softmax')(x)
    return models.Model(inputs, outputs)


def build_sac_critic():
    """
    Exact architecture from SAC.py build_critic():
    [state(4), action_onehot(4)] → Concat → Dense(128, relu) → Dense(128, relu) → Dense(1)
    """
    state_input = layers.Input(shape=(STATE_SIZE,))
    action_input = layers.Input(shape=(ACTION_SIZE,))
    x = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    q = layers.Dense(1)(x)
    return models.Model([state_input, action_input], q)


class SACAgent:
    """Exact match with SAC.py SACAgent class."""

    def __init__(self, lr=0.0003, mem_size=10000, tau=0.005):
        self.tau = tau

        # From SAC.py: actor + twin critics + twin target critics
        self.actor = build_sac_actor()
        self.critic1 = build_sac_critic()
        self.critic2 = build_sac_critic()
        self.target_critic1 = build_sac_critic()
        self.target_critic2 = build_sac_critic()
        self.update_targets(1.0)  # Hard copy initially (from SAC.py __init__)

        self.memory = deque(maxlen=mem_size)

        # From SAC.py: separate Adam optimizers
        self.actor_optimizer = optimizers.Adam(lr)
        self.critic1_optimizer = optimizers.Adam(lr)
        self.critic2_optimizer = optimizers.Adam(lr)

    def update_targets(self, tau_val=None):
        """From SAC.py: Polyak soft update of target critic networks."""
        if tau_val is None:
            tau_val = self.tau
        for target, source in zip(self.target_critic1.weights, self.critic1.weights):
            target.assign(tau_val * source + (1 - tau_val) * target)
        for target, source in zip(self.target_critic2.weights, self.critic2.weights):
            target.assign(tau_val * source + (1 - tau_val) * target)

    def get_action(self, state):
        """From SAC.py: sample action from actor's softmax distribution."""
        probs = self.actor(state.reshape(1, -1), training=False).numpy()[0]
        return int(np.random.choice(ACTIONS, p=probs))

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train(self, batch_size, gamma, alpha):
        """
        From SAC.py SACAgent.train():
        Twin critic update with GradientTape + actor update.
        Shapes corrected for proper broadcasting.
        """
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        states = np.array([b[0] for b in batch]).astype(np.float32)
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch]).astype(np.float32).reshape(-1, 1)
        next_states = np.array([b[3] for b in batch]).astype(np.float32)
        dones = np.array([b[4] for b in batch]).astype(np.float32).reshape(-1, 1)

        actions_onehot = tf.one_hot(actions, ACTION_SIZE)

        # === Critic update (from SAC.py) ===
        with tf.GradientTape(persistent=True) as tape:
            next_probs = self.actor(next_states)
            next_log_probs = tf.math.log(next_probs + 1e-10)

            q1_target = self.target_critic1([next_states, next_probs])
            q2_target = self.target_critic2([next_states, next_probs])
            q_target = tf.minimum(q1_target, q2_target)

            # Entropy: E_π[log π] = Σ_a π(a) * log π(a)
            entropy_term = tf.reduce_sum(next_probs * next_log_probs, axis=1, keepdims=True)
            target = rewards + gamma * (1.0 - dones) * (q_target - alpha * entropy_term)

            q1 = self.critic1([states, actions_onehot])
            q2 = self.critic2([states, actions_onehot])

            critic1_loss = tf.reduce_mean((q1 - target) ** 2)
            critic2_loss = tf.reduce_mean((q2 - target) ** 2)

        grads1 = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        grads2 = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        del tape

        self.critic1_optimizer.apply_gradients(zip(grads1, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(grads2, self.critic2.trainable_variables))

        # === Actor update (from SAC.py) ===
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            log_probs = tf.math.log(probs + 1e-10)

            q1 = self.critic1([states, probs])
            q2 = self.critic2([states, probs])
            q = tf.minimum(q1, q2)

            # Actor loss: maximize Q while maintaining entropy
            entropy_loss = tf.reduce_sum(probs * log_probs, axis=1, keepdims=True)
            actor_loss = tf.reduce_mean(alpha * entropy_loss - q)

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        # Soft update targets (from SAC.py)
        self.update_targets()


def run_sac(grid_size, obstacles, starts, goals):
    """
    Full SAC training + path generation.
    Exact algorithm structure from SAC.py, reduced episodes.
    """
    # Hyperparameters from SAC.py
    gamma = 0.99
    tau = 0.005
    alpha = 0.2   # entropy coefficient
    learning_rate = 0.0003
    batch_size = 64
    memory_size = 10000

    # Exact match with SAC.py ground truth
    EPISODES_TRAIN = 5000
    MAX_STEPS = 200

    num_agents = len(starts)
    env = DeepRLGridWorld(grid_size, obstacles, starts, goals)
    agents = [SACAgent(lr=learning_rate, mem_size=memory_size, tau=tau) for _ in range(num_agents)]

    # === TRAINING (exact structure from SAC.py training loop) ===
    for ep in range(EPISODES_TRAIN):
        states = env.reset()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            actions = [agents[i].get_action(states[i]) for i in range(num_agents)]
            next_states, collisions, done = env.step(actions)

            for i in range(num_agents):
                reward = compute_reward(env.positions[i], env.goals[i], collisions)
                agents[i].remember(states[i], actions[i], reward, next_states[i], done)
                agents[i].train(batch_size, gamma, alpha)

            states = next_states
            steps += 1

    # === TESTING (exact structure from SAC.py testing loop) ===
    # In testing, SAC uses deterministic argmax (from SAC.py testing)
    states = env.reset()
    paths = [[list(s)] for s in starts]
    done = False
    steps = 0
    collisions_count = 0

    while not done and steps < MAX_STEPS:
        actions = []
        for i in range(num_agents):
            # From SAC.py testing: argmax of actor probabilities
            probs = agents[i].actor(states[i].reshape(1, -1), training=False).numpy()[0]
            actions.append(int(np.argmax(probs)))

        next_states, collisions, done = env.step(actions)
        collisions_count += collisions

        for i in range(num_agents):
            paths[i].append(list(env.positions[i]))

        states = next_states
        steps += 1

    # Trim paths: stop each UAV at its goal
    goals_list = [list(g) for g in env.goals]
    paths = trim_paths_at_goals(paths, goals_list)

    return paths, done, steps, collisions_count


# ================================================================
#  UNIFIED PLAN FUNCTION
# ================================================================
def plan(grid_size, obstacles, starts, goals, mode="DQN"):
    start_time = time.perf_counter()

    if mode == "DQN":
        paths, done, steps, collisions_count = run_dqn(grid_size, obstacles, starts, goals)
    else:
        paths, done, steps, collisions_count = run_sac(grid_size, obstacles, starts, goals)

    end_time = time.perf_counter()
    compute_time_ms = (end_time - start_time) * 1000

    return {
        "paths": paths,
        "metrics": {
            "success": done,
            "steps": steps,
            "collisions": collisions_count,
            "time_ms": round(compute_time_ms, 3)
        }
    }
