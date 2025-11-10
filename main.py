from cvrp_instances import CVRPInstance
from aco_algo import AntColonyCVRP
from utils import display_results  # if saved in utils.py
import time

# Load instance
# just added these file_names so its easier to switch between instances
file_names = {
    "A-n32-k5": "/Users/alanale/Downloads/Vrp-Set-A/A-n32-k5.vrp", # best gap so far 3.98%
    "X-n106-k14": "/Users/alanale/Downloads/Vrp-Set-X/X/X-n106-k14.vrp", # best gap so far 4.78%
    "X-n101-k25": "/Users/alanale/Downloads/Vrp-Set-X/X/X-n101-k25.vrp", # best gap so far 8.51%
    "C101": "/Users/alanale/Downloads/Vrp-Set-Solomon/C101.txt", # best gap so far 3.86%
    "X-n200-k36": "/Users/alanale/Downloads/Vrp-Set-X/X/X-n200-k36.vrp"
}

instance_path = "/Users/alanale/Downloads/Vrp-Set-X/X/X-n106-k14.vrp"
# you can change the instance by changing the instance_path to any of the file_names values above
instance = CVRPInstance(instance_path)
# prints a summary of the instance such as # of customers, # vehicles, capacity, and name of the instance
instance.summary()

# Run ACO by creating an instance of AntColonyCVRP and providing parameters tuned for larger instances
aco = AntColonyCVRP(
    dist_matrix=instance.dist_matrix,
    demands=instance.demands,
    vehicle_capacity=instance.vehicle_capacity,
    num_vehicles=instance.num_vehicles,
    alpha=1.2,       # more pheromone influence (more exploitation)
    beta=2.0,        # slightly less greedy, more exploration
    rho=0.35,        # faster evaporation to escape local optima
    Q=100.0,
    num_ants=40,     # larger colony = better exploration
    max_iter=40,     # allow enough convergence
    local_search=True,
    rng_seed=42
)


# Measure execution time
start = time.time()
# Run the ACO algorithm where verbose is True to see progress
best_solution, best_cost = aco.run(verbose=True)
# Measure elapsed time
elapsed = time.time() - start

# Display formatted result summary
display_results(instance_path, best_cost, elapsed)

