import heapq
import time

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def astar_single(start, goal, obstacles, grid_size):
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

        for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + move[0], current[1] + move[1])

            if not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size):
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
                positions.append(p[-1])  # stay at goal

        # Count how many duplicate positions exist at time step t
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
        path = astar_single(start, goal, obstacles_set, grid_size)
        
        if path is None:
            success = False
            # Return partial path or empty path if not found
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
