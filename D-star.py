import numpy as np
import random
import heapq

import argparse
# =========================
# ARGUMENT PARSER
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--grid_size", type=int, default=10)
parser.add_argument("--num_agents", type=int, default=3)
parser.add_argument("--num_obstacles", type=int, default=20)

EPISODES_TEST = 1000

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

# =========================
# D* LITE COMPONENTS
# =========================
INF = float('inf')

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class DStarLite:
    def __init__(self, start, goal, obstacles):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles

        self.g = {}
        self.rhs = {}
        self.U = []

        self.km = 0

        self.rhs[self.goal] = 0
        self.g[self.goal] = INF

        heapq.heappush(self.U, (self.calculate_key(self.goal), self.goal))

    def calculate_key(self, s):
        g_rhs = min(self.g.get(s, INF), self.rhs.get(s, INF))
        return (g_rhs + heuristic(self.start, s) + self.km, g_rhs)

    def get_neighbors(self, s):
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        neighbors = []
        for m in moves:
            n = (s[0]+m[0], s[1]+m[1])
            if 0 <= n[0] < GRID_SIZE and 0 <= n[1] < GRID_SIZE:
                if n not in self.obstacles:
                    neighbors.append(n)
        return neighbors

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min([
                self.g.get(s, INF) + 1 for s in self.get_neighbors(u)
            ] + [INF])

        # remove u from queue if exists
        self.U = [(k, s) for k, s in self.U if s != u]
        heapq.heapify(self.U)

        if self.g.get(u, INF) != self.rhs.get(u, INF):
            heapq.heappush(self.U, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        while self.U:
            k_old, u = heapq.heappop(self.U)
            k_new = self.calculate_key(u)

            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
            elif self.g.get(u, INF) > self.rhs.get(u, INF):
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = INF
                for s in self.get_neighbors(u) + [u]:
                    self.update_vertex(s)

    def get_path(self):
        path = [self.start]
        current = self.start

        while current != self.goal:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                return None

            current = min(neighbors, key=lambda s: self.g.get(s, INF))
            path.append(current)

            if len(path) > GRID_SIZE * GRID_SIZE:
                return None

        return path

# =========================
# MULTI-AGENT SIMULATION
# =========================
def simulate(paths):
    max_len = max(len(p) for p in paths)
    collisions = 0

    for t in range(max_len):
        positions = []
        for p in paths:
            if t < len(p):
                positions.append(p[t])
            else:
                positions.append(p[-1])

        if len(set(positions)) < len(positions):
            collisions += 1

    return collisions

# =========================
# TESTING
# =========================
print("\n===== D* TESTING STARTED =====\n")

success_count = 0
total_steps = 0
total_collisions = 0

for ep in range(EPISODES_TEST):
    env = GridWorld()

    paths = []
    success = True

    for i in range(NUM_AGENTS):
        planner = DStarLite(env.starts[i], env.goals[i], env.obstacles)
        planner.compute_shortest_path()
        path = planner.get_path()

        if path is None:
            success = False
            break

        paths.append(path)

    if success:
        success_count += 1
        steps = max(len(p) for p in paths)
        collisions = simulate(paths)

        total_steps += steps
        total_collisions += collisions
    else:
        total_steps += GRID_SIZE * GRID_SIZE
        total_collisions += 10

    print(f"Episode: {ep+1}, Success: {int(success)}")

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