import numpy as np
import random
import argparse
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class RunConfig:
    grid_size: int
    num_uavs: int
    num_obstacles: int
    max_steps_per_episode: int = 100
    train_episodes: int = 5000
    test_episodes: int = 1000
    learning_rate: float = 1e-3
    device: str = "cpu"

    @property
    def action_dim(self):
        return 4

    @property
    def state_dim(self):
        return self.num_uavs * 6


# =========================
# ENVIRONMENT
# =========================
class GridEnv:
    def __init__(self, config):
        self.config = config
        self.grid_size = config.grid_size
        self.num_uavs = config.num_uavs
        self.num_obstacles = config.num_obstacles
        self.reset()

    def random_cell(self):
        return (random.randint(0, self.grid_size-1),
                random.randint(0, self.grid_size-1))

    def reset(self):
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:
            self.obstacles.add(self.random_cell())

        self.starts, self.goals = [], []

        for _ in range(self.num_uavs):
            while True:
                s = self.random_cell()
                if s not in self.obstacles:
                    break
            while True:
                g = self.random_cell()
                if g not in self.obstacles and g != s:
                    break
            self.starts.append(s)
            self.goals.append(g)

        self.pos = list(self.starts)
        return self.get_state()

    def get_state(self):
        state = []
        g = float(self.grid_size)

        for i in range(self.num_uavs):
            x, y = self.pos[i]
            gx, gy = self.goals[i]
            state.extend([x/g, y/g, gx/g, gy/g, (gx-x)/g, (gy-y)/g])

        return np.array(state, dtype=np.float32)

    def step(self, actions):
        new_pos = []
        collision = False

        for i, a in enumerate(actions):
            x, y = self.pos[i]

            if a == 0: x -= 1
            elif a == 1: x += 1
            elif a == 2: y -= 1
            elif a == 3: y += 1

            x = max(0, min(self.grid_size-1, x))
            y = max(0, min(self.grid_size-1, y))
            new_pos.append((x, y))

        # Check collisions
        for i, p in enumerate(new_pos):
            if p in self.obstacles:
                collision = True
            for j in range(len(new_pos)):
                if i != j and p == new_pos[j]:
                    collision = True

        self.pos = new_pos

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
# =========================
class Planner:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def next_pos(self, pos, action):
        x, y = pos
        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1

        x = max(0, min(self.grid_size-1, x))
        y = max(0, min(self.grid_size-1, y))
        return (x, y)

    def plan(self, env):
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
# =========================
class Actor(nn.Module):
    def __init__(self, state_dim, num_uavs, action_dim=4):
        super().__init__()
        self.num_uavs = num_uavs
        self.action_dim = action_dim

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
# TRAINING
# =========================
def train(config, env, planner, actor):
    optimizer = optim.Adam(actor.parameters(), lr=config.learning_rate)
    success_count = 0

    for ep in range(config.train_episodes):
        state = env.reset()
        collision_flag = False

        for step in range(config.max_steps_per_episode):
            actions = planner.plan(env)

            next_state, _, done, collision = env.step(actions)

            state_t = torch.tensor(state).float().unsqueeze(0)
            logits = actor.logits(state_t)

            target = torch.tensor(actions).unsqueeze(0)

            loss = nn.functional.cross_entropy(
                logits.view(-1, config.action_dim),
                target.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            collision_flag |= collision

            if done:
                break

        if done and not collision_flag:
            success_count += 1

        print(f"Episode {ep+1}, Success Rate: {100*success_count/(ep+1):.2f}%")

    return success_count / config.train_episodes


# =========================
# TESTING
# =========================
def evaluate(config, env, planner):
    success_count = 0

    for ep in range(config.test_episodes):
        state = env.reset()
        collision_flag = False

        for _ in range(config.max_steps_per_episode):
            actions = planner.plan(env)
            state, _, done, collision = env.step(actions)

            collision_flag |= collision

            if done:
                break

        if done and not collision_flag:
            success_count += 1

    return success_count / config.test_episodes


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Multi-UAV Path Planning")

    parser.add_argument("--grid_size", type=int, required=True)
    parser.add_argument("--num_uavs", type=int, required=True)
    parser.add_argument("--num_obstacles", type=int, required=True)

    args = parser.parse_args()

    config = RunConfig(
        grid_size=args.grid_size,
        num_uavs=args.num_uavs,
        num_obstacles=args.num_obstacles
    )

    print("\n===== CONFIGURATION =====")
    print(f"Grid Size: {config.grid_size}")
    print(f"UAVs: {config.num_uavs}")
    print(f"Obstacles: {config.num_obstacles}")

    env = GridEnv(config)
    planner = Planner(config.grid_size)
    actor = Actor(config.state_dim, config.num_uavs)

    print("\n===== TRAINING =====")
    train_rate = train(config, env, planner, actor)

    print("\n===== TESTING =====")
    test_rate = evaluate(config, env, planner)

    print("\n===== RESULTS =====")
    print(f"Train Success Rate: {train_rate*100:.2f}%")
    print(f"Test Success Rate: {test_rate*100:.2f}%")


if __name__ == "__main__":
    main()