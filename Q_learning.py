import numpy as np
import random
import argparse
# =========================
# ARGUMENT PARSER
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--grid_size", type=int, default=10)
parser.add_argument("--num_agents", type=int, default=3)
parser.add_argument("--num_obstacles", type=int, default=20)

#args = parser.parse_args()

#GRID_SIZE = args.grid_size
#NUM_AGENTS = args.num_agents
#NUM_OBSTACLES = args.num_obstacles

EPISODES_TRAIN = 5000
EPISODES_TEST = 1000
MAX_STEPS = 200

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_MAP = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# =========================
# HYPERPARAMETERS
# =========================
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

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
        return self.positions

    def step(self, actions):
        new_positions = []
        collisions = 0

        for i, action in enumerate(actions):
            move = ACTION_MAP[action]
            new_pos = (self.positions[i][0] + move[0],
                       self.positions[i][1] + move[1])

            # Boundary check
            if not (0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE):
                new_pos = self.positions[i]

            # Obstacle collision
            if new_pos in self.obstacles:
                new_pos = self.positions[i]
                collisions += 1

            new_positions.append(new_pos)

        # Inter-agent collision
        if len(set(new_positions)) < len(new_positions):
            collisions += 1

        self.positions = new_positions

        done = all(self.positions[i] == self.goals[i] for i in range(NUM_AGENTS))

        return self.positions, collisions, done

# =========================
# Q-TABLES
# =========================
Q_tables = [dict() for _ in range(NUM_AGENTS)]

def get_Q(agent, state):
    if state not in Q_tables[agent]:
        Q_tables[agent][state] = np.zeros(len(ACTIONS))
    return Q_tables[agent][state]

def choose_action(agent, state):
    if random.random() < epsilon:
        return random.randint(0, len(ACTIONS)-1)
    return np.argmax(get_Q(agent, state))

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
        actions_idx = []
        actions = []

        for i in range(NUM_AGENTS):
            action_idx = choose_action(i, states[i])
            actions_idx.append(action_idx)
            actions.append(ACTIONS[action_idx])

        next_states, collisions, done = env.step(actions)

        for i in range(NUM_AGENTS):
            reward = compute_reward(next_states[i], env.goals[i], collisions)

            q_current = get_Q(i, states[i])[actions_idx[i]]
            q_next = np.max(get_Q(i, next_states[i]))

            Q_tables[i][states[i]][actions_idx[i]] += \
                alpha * (reward + gamma * q_next - q_current)

        states = next_states
        steps += 1

    # Success tracking
    if done:
        success_train += 1
        success_flag = 1
    else:
        success_flag = 0

    running_success_rate = (success_train / (ep + 1)) * 100

    # Print every episode
    print(f"Episode: {ep+1}, Steps: {steps}, Success: {success_flag}, Running Success Rate: {running_success_rate:.2f}%")

    # Epsilon decay
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

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
            action = np.argmax(get_Q(i, states[i]))
            actions.append(ACTIONS[action])

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