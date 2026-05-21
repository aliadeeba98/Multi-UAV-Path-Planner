from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import random
from typing import List, Tuple

from app.algorithms import plan_paths

app = FastAPI(title="TRAVERSE: Multi-UAV Path Planner")

# Pydantic models for request validation
class PlanRequest(BaseModel):
    grid_size: int
    obstacles: List[List[int]]
    starts: List[List[int]]
    goals: List[List[int]]
    algorithm: str

# Preset maps generator
def generate_preset_maps():
    maps = {}
    
    # 1. Empty 10x10
    # Goals start on top of their own UAV — user relocates via Place Goal tool
    empty_starts = [[0, 0], [0, 9]]
    maps["empty-10-10"] = {
        "grid_size": 10,
        "obstacles": [],
        "starts": empty_starts,
        "goals": [s[:] for s in empty_starts]
    }
    
    # 2. Random 10x10
    # Let's seed or use a fixed random layout for repeatability
    random_obs = []
    random.seed(42)
    random_starts = [[1, 1], [1, 8], [8, 1]]
    reserved = {(s[0], s[1]) for s in random_starts}
    while len(random_obs) < 18:
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        if (x, y) not in reserved and [x, y] not in random_obs:
            random_obs.append([x, y])
            
    maps["random-10-10"] = {
        "grid_size": 10,
        "obstacles": random_obs,
        "starts": random_starts,
        "goals": [s[:] for s in random_starts]
    }
    
    # 3. Warehouse 20x20
    # Layout representing warehouse racks
    warehouse_obs = []
    for x in [3, 4, 7, 8, 11, 12, 15, 16]:
        for y in range(2, 8):
            warehouse_obs.append([x, y])
        for y in range(12, 18):
            warehouse_obs.append([x, y])
    
    warehouse_starts = [[0, 0], [0, 19], [19, 0], [19, 19]]
    maps["warehouse-20-20"] = {
        "grid_size": 20,
        "obstacles": warehouse_obs,
        "starts": warehouse_starts,
        "goals": [s[:] for s in warehouse_starts]
    }
    
    # 4. Rooms Grid 32x32
    # 4 rooms separated by doors
    rooms_obs = []
    # Vertical wall
    for y in range(32):
        if y not in [7, 8, 23, 24]:
            rooms_obs.append([15, y])
    # Horizontal wall
    for x in range(32):
        if x not in [7, 8, 23, 24]:
            rooms_obs.append([x, 15])
    
    room_starts = [[2, 2], [2, 29], [29, 2], [29, 29]]
    maps["room-32-32-4"] = {
        "grid_size": 32,
        "obstacles": rooms_obs,
        "starts": room_starts,
        "goals": [s[:] for s in room_starts]
    }
    
    return maps

PRESET_MAPS = generate_preset_maps()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend templates/index.html not found.")

@app.get("/api/maps")
async def get_maps():
    return PRESET_MAPS

@app.post("/api/plan")
async def execute_plan(request: PlanRequest):
    try:
        result = plan_paths(
            grid_size=request.grid_size,
            obstacles=request.obstacles,
            starts=request.starts,
            goals=request.goals,
            algorithm=request.algorithm
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Planning execution error: {str(e)}")
