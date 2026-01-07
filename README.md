# MGCBS Warehouse Simulation

This project implements a Multi-Goal Conflict-Based Search (MGCBS) variant A2 algorithm from Tang et al. (2024) article namely "MGCBS:AnOptimal and Efficient Algorithm for Solving Multi-Goal Multi-Agent Path Finding Problem". This algorithm used for multi-agent pathfinding in a warehouse environment. It is designed to simulate agents (e.g., robots or workers) navigating a grid-based warehouse to complete tasks involving picking items from shelves and delivering to loading stations, while avoiding conflicts.

The code is part of a thesis (skripsi) on implementing multi-agent systems into mock warehouse automation.

## Features

- **Warehouse Layout**: Customizable grid with walls, shelves, loading stations, docking stations, narrow aisles, and intersections.
- **Multi-Agent Tasks**: Agents have multiple goals (shelves to visit) and a final loading station.
- **Conflict Resolution**: Uses CBS (Conflict-Based Search) with low-level MGCBS for multi-goal planning.
- **Visualization**: Generates per-timestep visualizations of agent paths.
- **Metrics**: Tracks search performance (nodes expanded, conflicts, etc.).
- **Experiments**: Supports running single scenarios or batch experiments with varying parameters.

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib

## Usage

### Running a Single Scenario

To run a single experiment scenario, modify the parameters in `main.py` and call `run_single_scenario()`.

Example in `main.py`:

```python
# Example: Run a scenario with 4 agents, each with 2 tasks, seed 42
result = run_single_scenario(
    num_agents=4,
    tasks_per_agent=2,
    seed=42,
    verbose=True,
    time_limit=30.0
)
```

- `num_agents`: Number of agents (e.g., 2, 4, 6, 8).
- `tasks_per_agent`: Number of shelves each agent must visit (e.g., 1, 2, 3, 5).
- `seed`: Random seed for reproducibility.
- `verbose`: Print detailed output.
- `time_limit`: Maximum time in seconds for the solver.

The function will output results to the console and save detailed JSON and summary CSV files directly in the same directory, as for visualization it will saved in a directory like `na{num_agents}_nt{tasks_per_agent}_seed{seed}/`.

### Running All Experiments

To run a batch of experiments, use `run_all_experiments(iterations=3)` at the end of `main.py`.

### Customizing the Warehouse

Edit `create_warehouse_grid()` in `main.py` to change the grid size, walls, shelves, stations, etc.

### Visualization

After running a scenario, visualizations are saved in subfolders (e.g., `iter1/na4_nt2_seed1042/`) as PNG images showing agent positions per timestep.

Use `visualize_paths_per_timestep()` to generate custom visualizations.

## Output Files

- **JSON**: Detailed results including paths, metrics, and solver status.
- **CSV**: Summary with key metrics (success rate, time, nodes expanded).
- **Images**: Per-timestep path visualizations.
- **Text**: Agent paths report in some experiments.

## Key Modules

- `main.py`: Entry point, grid creation, agent generation, experiment runners.
- `grid.py`: Grid map class with warehouse elements.
- `agent.py`: Agent task definitions.
- `high_level_mgcbs_a2.py`: High-level CBS solver.
- `low_level_mgcbs_a2.py`: Low-level multi-goal planner.
- `low_level_astar.py`: A* search for intervals.
- `constraints.py`: Constraint handling.
- `intervals.py`: Safe intervals for goals.
- `heuristics.py`: Distance tables and MST heuristics.
- `visualization.py`: Plotting functions.
- `metrics.py`: Performance tracking.

## Troubleshooting

- If no solution is found, increase `time_limit` or adjust `time_horizon`.
- Ensure the grid has enough space for agents to navigate without deadlocks.
- For large grids/agents, the solver may timeout; monitor metrics.