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
# A* ALGORITHM
# =========================
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def astar(start, goal, obstacles):
    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_cost = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for move in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+move[0], current[1]+move[1])

            if not (0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE):
                continue

            if neighbor in obstacles:
                continue

            new_cost = g_cost[current] + 1

            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current

    return None  # No path found

# =========================
# MULTI-AGENT EXECUTION
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
                positions.append(p[-1])  # stay at goal

        if len(set(positions)) < len(positions):
            collisions += 1

    return collisions

# =========================
# TESTING (NO TRAINING)
# =========================
print("\n===== A* TESTING STARTED =====\n")

success_count = 0
total_steps = 0
total_collisions = 0

for ep in range(EPISODES_TEST):
    env = GridWorld()

    paths = []
    success = True

    for i in range(NUM_AGENTS):
        path = astar(env.starts[i], env.goals[i], env.obstacles)

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
        total_steps += GRID_SIZE * GRID_SIZE  # penalty
        total_collisions += 10  # penalty

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