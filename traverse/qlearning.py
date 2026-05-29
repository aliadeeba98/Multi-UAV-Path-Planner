import numpy as np
import random
import time

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_MAP = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

class QLearningGridWorld:
    def __init__(self, grid_size, obstacles, starts, goals):
        self.grid_size = grid_size
        self.obstacles = set(tuple(o) for o in obstacles)
        self.starts = [tuple(s) for s in starts]
        self.goals = [tuple(g) for g in goals]
        self.num_agents = len(starts)
        self.reset()

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
            if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
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
        done = all(self.positions[i] == self.goals[i] for i in range(self.num_agents))

        return self.positions, collisions, done

def compute_reward(pos, goal, collision):
    if pos == goal:
        return 100
    if collision:
        return -10
    return -1

def train_and_plan(grid_size, obstacles, starts, goals):
    start_time = time.perf_counter()

    num_agents = len(starts)
    env = QLearningGridWorld(grid_size, obstacles, starts, goals)

    # Q-tables: dictionaries mapping (state) to an array of size 4
    Q_tables = [dict() for _ in range(num_agents)]

    def get_Q(agent, state):
        if state not in Q_tables[agent]:
            Q_tables[agent][state] = np.zeros(len(ACTIONS))
        return Q_tables[agent][state]

    alpha = 0.2
    gamma = 0.95
    epsilon = 0.8
    epsilon_decay = 0.985
    epsilon_min = 0.05
    episodes = 250
    max_steps = grid_size * 4

    # Run quick training
    for ep in range(episodes):
        states = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            actions_idx = []
            actions = []

            for i in range(num_agents):
                # Epsilon-greedy action choice
                if random.random() < epsilon:
                    action_idx = random.randint(0, len(ACTIONS) - 1)
                else:
                    action_idx = np.argmax(get_Q(i, states[i]))
                actions_idx.append(action_idx)
                actions.append(ACTIONS[action_idx])

            next_states, collisions, done = env.step(actions)

            for i in range(num_agents):
                reward = compute_reward(next_states[i], env.goals[i], collisions > 0)
                q_current = get_Q(i, states[i])[actions_idx[i]]
                q_next = np.max(get_Q(i, next_states[i]))

                # Temporal Difference update
                Q_tables[i][states[i]][actions_idx[i]] += \
                    alpha * (reward + gamma * q_next - q_current)

            states = next_states
            steps += 1

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    # testing phase (generate the path)
    states = env.reset()
    done = False
    steps = 0
    collisions_episode = 0

    paths = [[s] for s in starts]

    while not done and steps < max_steps:
        actions = []
        for i in range(num_agents):
            action_idx = np.argmax(get_Q(i, states[i]))
            actions.append(ACTIONS[action_idx])

        next_states, collisions, done = env.step(actions)
        collisions_episode += collisions

        for i in range(num_agents):
            paths[i].append(next_states[i])

        states = next_states
        steps += 1

    end_time = time.perf_counter()
    compute_time_ms = (end_time - start_time) * 1000

    # If agents get stuck (no progress), success is false
    success = done

    # Trim paths: stop each UAV at the first time it reaches its goal
    goals_list = [list(g) for g in env.goals]
    trimmed_paths = []
    for i, path in enumerate(paths):
        goal = goals_list[i]
        trimmed = []
        for pos in path:
            p = list(pos) if not isinstance(pos, list) else pos
            trimmed.append(p)
            if p[0] == goal[0] and p[1] == goal[1]:
                break
        trimmed_paths.append(trimmed)

    return {
        "paths": trimmed_paths,
        "metrics": {
            "success": success,
            "steps": steps,
            "collisions": collisions_episode,
            "time_ms": round(compute_time_ms, 3)
        }
    }
