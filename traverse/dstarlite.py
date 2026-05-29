import heapq
import time

INF = float('inf')

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class DStarLiteSolver:
    def __init__(self, start, goal, obstacles, grid_size):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.grid_size = grid_size

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
            if 0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size:
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

            if len(path) > self.grid_size * self.grid_size:
                return None

        return path

def simulate_collisions(paths):
    if not paths:
        return 0
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

def plan(grid_size, obstacles, starts, goals):
    start_time = time.perf_counter()
    paths = []
    success = True

    obstacles_set = set(tuple(o) for o in obstacles)

    for i in range(len(starts)):
        start = tuple(starts[i])
        goal = tuple(goals[i])
        planner = DStarLiteSolver(start, goal, obstacles_set, grid_size)
        planner.compute_shortest_path()
        path = planner.get_path()

        if path is None:
            success = False
            paths.append([start])
        else:
            paths.append(path)

    end_time = time.perf_counter()
    compute_time_ms = (end_time - start_time) * 1000

    if success:
        steps = max(len(p) for p in paths) - 1
        collisions = simulate_collisions(paths)
    else:
        steps = grid_size * grid_size
        collisions = 10

    return {
        "paths": paths,
        "metrics": {
            "success": success,
            "steps": max(0, steps),
            "collisions": collisions,
            "time_ms": round(compute_time_ms, 3)
        }
    }
