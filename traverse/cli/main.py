import argparse
import sys
import time
from traverse.algorithms import plan_paths

def generate_random_setup(grid_size, num_agents, num_obstacles):
    import random
    random.seed(int(time.time()))
    
    obstacles = set()
    max_cells = grid_size * grid_size
    # Bound obstacles to 40% of grid to keep it solvable
    safe_num_obstacles = min(num_obstacles, int(max_cells * 0.4))
    
    while len(obstacles) < safe_num_obstacles:
        obstacles.add((random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)))
        
    starts = []
    # Avoid infinite loops in dense grids
    while len(starts) < num_agents and len(starts) + len(obstacles) < max_cells:
        pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if pos not in obstacles and pos not in starts:
            starts.append(pos)
            
    goals = []
    while len(goals) < num_agents and len(starts) + len(obstacles) + len(goals) < max_cells:
        pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if pos not in obstacles and pos not in starts and pos not in goals:
            goals.append(pos)
            
    return list(list(o) for o in obstacles), list(list(s) for s in starts), list(list(g) for g in goals)

def handle_gui(args):
    from traverse.main import run_server
    run_server(host=args.host, port=args.port)

def safe_print(text, *args, **kwargs):
    replacements = {
        "🗺️": "",
        "🔄": "",
        "⚠️": "WARNING:",
        "🚀": "",
        "✨": "",
        "❌": "",
        "🏆": "",
        "⏱️": "",
        "🏁": "",
        "💥": "",
        "📈": "",
        "🛸": "",
        "➔": "->"
    }
    try:
        text.encode(sys.stdout.encoding or 'utf-8')
        print(text, *args, **kwargs)
    except UnicodeEncodeError:
        text_clean = text
        for k, v in replacements.items():
            text_clean = text_clean.replace(k, v)
        try:
            print(text_clean.encode(sys.stdout.encoding or 'ascii', errors='ignore').decode(sys.stdout.encoding or 'ascii'), *args, **kwargs)
        except Exception:
            print(text_clean.encode('ascii', errors='ignore').decode('ascii'), *args, **kwargs)

def handle_plan(args):
    safe_print("=" * 60)
    safe_print("🗺️   TRAVERSE MULTI-UAV PATH PLANNER CLI")
    safe_print("=" * 60)
    safe_print(f"Algorithm:      {args.algo.upper()}")
    safe_print(f"Grid Size:      {args.grid_size} x {args.grid_size}")
    safe_print(f"UAV Agents:     {args.num_agents}")
    safe_print(f"Obstacles:      {args.num_obstacles}")
    safe_print("-" * 60)
    safe_print("🔄 Generating random environment layout...")
    
    obstacles, starts, goals = generate_random_setup(args.grid_size, args.num_agents, args.num_obstacles)
    
    if len(starts) < args.num_agents:
        safe_print("⚠️  Warning: Unable to fit all requested agents in grid without spawning on obstacles.")
        safe_print(f"Running simulation with {len(starts)} agents instead.")
        
    safe_print(f"🚀 Launching {args.algo.upper()} path planning execution solver...")
    safe_print("-" * 60)
    
    start_time = time.perf_counter()
    result = plan_paths(args.grid_size, obstacles, starts, goals, args.algo)
    elapsed = (time.perf_counter() - start_time) * 1000
    
    metrics = result["metrics"]
    paths = result["paths"]
    
    status = "✨ SUCCESS (Goal Reached)" if metrics["success"] else "❌ FAILED (Goal Unreachable / Penalty Applied)"
    
    safe_print("================== PLANNING RESULTS ==================")
    safe_print(f"🏆 Status:          {status}")
    safe_print(f"⏱️  Compute Time:    {metrics['time_ms']:.3f} ms (CLI process: {elapsed:.2f} ms)")
    safe_print(f"🏁 Total Steps:     {metrics['steps']}")
    safe_print(f"💥 Collisions:      {metrics['collisions']}")
    safe_print("======================================================")
    
    if metrics["success"]:
        safe_print("\n📈 Calculated UAV Paths:")
        for idx, path in enumerate(paths):
            safe_print(f"  🛸 UAV {idx + 1}: {path[0]} ➔ {path[-1]} (length: {len(path)} steps)")
            if len(path) <= 10:
                safe_print(f"     Path: {path}")
            else:
                safe_print(f"     Path: {path[:4]} ... {path[-4:]}")
    safe_print("-" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="TRAVERSE: Multi-UAV Path Planner Package CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  traverse gui --port 8000
  traverse plan --algo astar --grid-size 15 --num-agents 4 --num-obstacles 25
  traverse plan --algo qlearning --grid-size 10 --num-agents 2
"""
    )
    
    subparsers = parser.add_subparsers(title="subcommands", dest="command", required=True)
    
    # GUI subcommand
    gui_parser = subparsers.add_parser("gui", help="Launch the web-based interactive path planning dashboard")
    gui_parser.add_argument("--host", type=str, default="127.0.0.1", help="Binding host address (default: 127.0.0.1)")
    gui_parser.add_argument("--port", type=int, default=8000, help="Port to run the uvicorn server on (default: 8000)")
    gui_parser.set_defaults(func=handle_gui)
    
    # Plan subcommand
    plan_parser = subparsers.add_parser("plan", help="Execute an algorithm planning run in the terminal")
    plan_parser.add_argument("--algo", type=str, default="astar", 
                            choices=["astar", "dstarlite", "dstar", "qlearning", "dqn", "sac", "pso", "psosac", "hybridpsosac"],
                            help="Algorithm to execute for path planning")
    plan_parser.add_argument("--grid-size", type=int, default=10, help="Size of the grid world square (default: 10)")
    plan_parser.add_argument("--num-agents", type=int, default=3, help="Number of UAV agents (default: 3)")
    plan_parser.add_argument("--num-obstacles", type=int, default=20, help="Number of static obstacles (default: 20)")
    plan_parser.set_defaults(func=handle_plan)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
