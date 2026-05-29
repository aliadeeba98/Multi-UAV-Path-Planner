import argparse
import sys
from traverse.qlearning import train_and_plan
from traverse.cli.main import generate_random_setup

def main():
    parser = argparse.ArgumentParser(description="Run Q-Learning Multi-Agent Path Planning")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size (default: 10)")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents (default: 3)")
    parser.add_argument("--num_obstacles", type=int, default=20, help="Number of static obstacles (default: 20)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes to run (each runs training + testing, default: 10 for speed)")
    args = parser.parse_args()

    print("\n===== Q-LEARNING EVALUATION STARTED =====\n")
    
    success_count = 0
    total_steps = 0
    total_collisions = 0
    
    for ep in range(args.episodes):
        obstacles, starts, goals = generate_random_setup(args.grid_size, args.num_agents, args.num_obstacles)
        
        # Run package Q-learning training and planning solver
        result = train_and_plan(args.grid_size, obstacles, starts, goals)
        metrics = result["metrics"]
        
        success = metrics["success"]
        if success:
            success_count += 1
            total_steps += metrics["steps"]
            total_collisions += metrics["collisions"]
        else:
            total_steps += args.grid_size * 4  # max steps limit penalty
            total_collisions += metrics["collisions"]
            
        print(f"Episode: {ep+1}, Success: {int(success)}")
        
    success_rate = (success_count / args.episodes) * 100
    avg_steps = total_steps / args.episodes
    avg_collisions = total_collisions / args.episodes
    
    print("\n===== FINAL RESULTS =====\n")
    print(f"Grid Size: {args.grid_size}")
    print(f"Agents: {args.num_agents}")
    print(f"Number of Obstacles: {args.num_obstacles}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Collisions: {avg_collisions:.2f}")

if __name__ == "__main__":
    main()