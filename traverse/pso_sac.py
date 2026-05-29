"""
Hybrid PSO-SAC implementation using PyTorch.
Exact algorithm match with ground-truth Hybrid_PSO_SAC.py.

Environment (GridEnv), Planner, and Actor network architectures
are all identical to the ground truth. Training uses cross-entropy
imitation learning from Planner demonstrations, matching the
ground-truth train() function.
"""
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# ENVIRONMENT
# Exact match with Hybrid_PSO_SAC.py GridEnv
# =========================
class PSOEnvironment:
    def __init__(self, grid_size, obstacles, starts, goals):
        self.grid_size = grid_size
        self.obstacles = set(tuple(o) for o in obstacles)
        self.starts = [tuple(s) for s in starts]
        self.goals = [tuple(g) for g in goals]
        self.num_uavs = len(starts)
        self.reset()

    def random_cell(self):
        return (random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1))

    def reset(self):
        self.pos = list(self.starts)
        return self.get_state()

    def get_state(self):
        """
        From Hybrid_PSO_SAC.py GridEnv.get_state():
        6 elements per UAV: [x/g, y/g, gx/g, gy/g, (gx-x)/g, (gy-y)/g]
        """
        state = []
        g = float(self.grid_size)
        for i in range(self.num_uavs):
            x, y = self.pos[i]
            gx, gy = self.goals[i]
            state.extend([x / g, y / g, gx / g, gy / g, (gx - x) / g, (gy - y) / g])
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        """
        Exact step logic from Hybrid_PSO_SAC.py GridEnv.step().
        Note: positions are CLAMPED to grid bounds (not reverted on obstacle).
        """
        new_pos = []
        collision = False

        for i, a in enumerate(actions):
            x, y = self.pos[i]
            if a == 0: x -= 1
            elif a == 1: x += 1
            elif a == 2: y -= 1
            elif a == 3: y += 1

            x = max(0, min(self.grid_size - 1, x))
            y = max(0, min(self.grid_size - 1, y))
            new_pos.append((x, y))

        # Check collisions (from Hybrid_PSO_SAC.py)
        for i, p in enumerate(new_pos):
            if p in self.obstacles:
                collision = True
            for j in range(len(new_pos)):
                if i != j and p == new_pos[j]:
                    collision = True

        self.pos = new_pos

        # Reward and done check (from Hybrid_PSO_SAC.py)
        rewards = []
        done = True
        for i in range(self.num_uavs):
            dist = np.linalg.norm(np.array(self.pos[i]) - np.array(self.goals[i]))
            r = -0.01 * dist
            if dist <= 1:
                r += 100
            else:
                done = False
            if collision:
                r -= 20
            rewards.append(r)

        return self.get_state(), sum(rewards), done, collision


# =========================
# PLANNER (Greedy baseline)
# Exact match with Hybrid_PSO_SAC.py Planner
# =========================
class PSOPlanner:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def next_pos(self, pos, action):
        """From Hybrid_PSO_SAC.py Planner.next_pos()"""
        x, y = pos
        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1
        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))
        return (x, y)

    def plan(self, env):
        """
        From Hybrid_PSO_SAC.py Planner.plan():
        Greedy action selection — pick action minimizing Euclidean distance to goal.
        """
        actions = []
        for i in range(env.num_uavs):
            best_action = 0
            best_dist = 1e9
            for a in range(4):
                p = self.next_pos(env.pos[i], a)
                if p in env.obstacles:
                    continue
                dist = np.linalg.norm(np.array(p) - np.array(env.goals[i]))
                if dist < best_dist:
                    best_dist = dist
                    best_action = a
            actions.append(best_action)
        return actions


# =========================
# ACTOR NETWORK
# Exact match with Hybrid_PSO_SAC.py Actor
# =========================
class Actor(nn.Module):
    def __init__(self, state_dim, num_uavs, action_dim=4):
        super().__init__()
        self.num_uavs = num_uavs
        self.action_dim = action_dim

        # Exact architecture from Hybrid_PSO_SAC.py
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_uavs * action_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, self.num_uavs, self.action_dim)
        return torch.softmax(x, dim=2)

    def logits(self, x):
        return self.net(x).view(-1, self.num_uavs, self.action_dim)


# =========================
# PATH TRIMMING
# =========================
def trim_paths_at_goals(paths, goals):
    """Stop each UAV's path at the first time it reaches its goal."""
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


# =========================
# PLAN FUNCTION
# =========================
def plan(grid_size, obstacles, starts, goals):
    start_time = time.perf_counter()

    num_uavs = len(starts)
    state_dim = num_uavs * 6  # From Hybrid_PSO_SAC.py RunConfig.state_dim

    env = PSOEnvironment(grid_size, obstacles, starts, goals)
    planner = PSOPlanner(grid_size)
    actor = Actor(state_dim, num_uavs)

    # Exact match with Hybrid_PSO_SAC.py ground truth
    learning_rate = 1e-3
    TRAIN_EPISODES = 5000
    MAX_STEPS = 200

    # ========================================
    # TRAINING (exact structure from Hybrid_PSO_SAC.py train())
    # Actor learns to imitate Planner via cross-entropy loss
    # ========================================
    optimizer = optim.Adam(actor.parameters(), lr=learning_rate)

    for ep in range(TRAIN_EPISODES):
        state = env.reset()
        collision_flag = False

        for step in range(MAX_STEPS):
            # Get planner's demonstration actions
            actions = planner.plan(env)

            next_state, _, done, collision = env.step(actions)

            # Train actor to imitate planner (from Hybrid_PSO_SAC.py train())
            state_t = torch.tensor(state).float().unsqueeze(0)
            logits = actor.logits(state_t)
            target = torch.tensor(actions).unsqueeze(0)

            loss = nn.functional.cross_entropy(
                logits.view(-1, 4),
                target.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            collision_flag |= collision

            if done:
                break

    # ========================================
    # PATH GENERATION — Use trained Actor for inference
    # ========================================
    state = env.reset()
    paths = [[list(s)] for s in starts]
    collision_count = 0
    step_idx = 0
    done = False

    # Track visited positions to prevent oscillation
    visited_counts = [dict() for _ in range(num_uavs)]
    arrived = [False] * num_uavs

    while not done and step_idx < MAX_STEPS:
        with torch.no_grad():
            state_t = torch.tensor(state).float().unsqueeze(0)
            probs = actor(state_t)[0]  # (num_uavs, 4)

        actions = []
        for i in range(num_uavs):
            if arrived[i]:
                # UAV already at goal — force stay by picking action
                # that keeps it at current position (towards goal = stay)
                actions.append(0)  # Placeholder, position will be overridden
                continue

            action_probs = probs[i].numpy().copy()
            current_pos = env.pos[i]

            # Track visits for anti-oscillation
            pos_key = current_pos
            visited_counts[i][pos_key] = visited_counts[i].get(pos_key, 0) + 1

            # Anti-oscillation: penalize frequently visited positions
            if visited_counts[i][pos_key] > 3:
                for a in range(4):
                    next_p = planner.next_pos(current_pos, a)
                    visit_count = visited_counts[i].get(next_p, 0)
                    if visit_count > 2:
                        action_probs[a] *= 0.01

                total = action_probs.sum()
                if total > 1e-8:
                    action_probs /= total
                else:
                    action_probs = np.ones(4) / 4.0

            action = int(np.argmax(action_probs))
            actions.append(action)

        _, _, done, collision = env.step(actions)
        state = env.get_state()

        # Override: arrived UAVs stay at goal
        for i in range(num_uavs):
            if arrived[i]:
                env.pos[i] = env.goals[i]

        if collision:
            collision_count += 1

        for i in range(num_uavs):
            paths[i].append(list(env.pos[i]))
            if env.pos[i] == env.goals[i]:
                arrived[i] = True

        # Check if all arrived
        if all(arrived):
            done = True

        step_idx += 1

    # Trim paths at goals
    goals_list = [list(g) for g in env.goals]
    paths = trim_paths_at_goals(paths, goals_list)

    end_time = time.perf_counter()
    compute_time_ms = (end_time - start_time) * 1000

    return {
        "paths": paths,
        "metrics": {
            "success": done,
            "steps": step_idx,
            "collisions": collision_count,
            "time_ms": round(compute_time_ms, 3)
        }
    }
