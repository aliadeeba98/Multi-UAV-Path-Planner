from app import astar
from app import dstarlite
from app import qlearning
from app import pso_sac
from app import deep_rl

def plan_paths(grid_size, obstacles, starts, goals, algorithm):
    """
    Unified entrypoint mapping incoming client requests to exact algorithm executions.
    All return formats are standardized:
    {
        "paths": [[(x,y), ...], ...],
        "metrics": {
            "success": bool,
            "steps": int,
            "collisions": int,
            "time_ms": float
        }
    }
    """
    algo = algorithm.lower().replace("-", "").replace("_", "")

    if algo == "astar":
        return astar.plan(grid_size, obstacles, starts, goals)
    elif algo == "dstarlite" or algo == "dstar":
        return dstarlite.plan(grid_size, obstacles, starts, goals)
    elif algo == "qlearning":
        return qlearning.train_and_plan(grid_size, obstacles, starts, goals)
    elif algo == "dqn":
        return deep_rl.plan(grid_size, obstacles, starts, goals, mode="DQN")
    elif algo == "sac":
        return deep_rl.plan(grid_size, obstacles, starts, goals, mode="SAC")
    elif algo == "pso" or algo == "psosac" or algo == "hybridpsosac":
        return pso_sac.plan(grid_size, obstacles, starts, goals)
    else:
        # Fallback to A*
        return astar.plan(grid_size, obstacles, starts, goals)
