from cvrp_instances import CVRPInstance
from aco_algo import AntColonyCVRP
from utils import display_results  # if saved in utils.py
import time
from CW import *

# Load instance
# just added these file_names so its easier to switch between instances
file_names = {
    "A-n32-k5": "/Users/alanale/Downloads/Vrp-Set-A/A-n32-k5.vrp", # best gap so far 3.98%
    "X-n106-k14": "/Users/alanale/Downloads/Vrp-Set-X/X/X-n106-k14.vrp", # best gap so far 4.78%
    "X-n101-k25": "/Users/alanale/Downloads/Vrp-Set-X/X/X-n101-k25.vrp", # best gap so far 8.51%
    "C101": "/Users/alanale/Downloads/Vrp-Set-Solomon/C101.txt", # best gap so far 3.86%
    "X-n200-k36": "/Users/alanale/Downloads/Vrp-Set-X/X/X-n200-k36.vrp"
}



# you can change the instance by changing the instance_path to any of the file_names values above
instance = CVRPInstance(INSTANCE_PATH)
# prints a summary of the instance such as # of customers, # vehicles, capacity, and name of the instance
instance.summary()

# Run ACO by creating an instance of AntColonyCVRP and providing parameters tuned for larger instances
aco_params = {
        "dist_matrix": EDGE_WEIGHT,
        "demands": demand,
        "vehicle_capacity": capacity,
        "alpha": 3.0,
        "beta": 1.0,
        "rho": 0.15,
        "Q": 100,
        "num_ants": 30,
        "max_iter": 20,
        "local_search": True,
        "seed_strength": 300.0,
        "elitist_weight": 5.0,
        "start_with_cw_frac": 0.3,
        "candidate_list_size": 20,
        "tau_min": 1e-6,
        "tau_max": 1e3
    }
    
aco = AntColonyCVRP(
    **aco_params
)
aco.cw_solution = clark_and_wright()  # ta fonction C&W


# Measure execution time
start = time.time()
# Run the ACO algorithm where verbose is True to see progress
best_solution, best_cost = aco.run(verbose=True)
# Measure elapsed time
elapsed = time.time() - start

# Display formatted result summary
display_results(INSTANCE_PATH, best_cost, elapsed)
show(best_solution,data_set)
show(BEST_SOLUTION,data_set)

