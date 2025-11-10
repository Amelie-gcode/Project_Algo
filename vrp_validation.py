# run_vrp_validation.py
import time
import vrplib
from aco_algo import AntColonyCVRP
from cvrp_instances import CVRPInstance  # your classes from the code above

#IGNORE THIS FILE FOR NOW THIS IS JUST THE MANDITORY VRPLIB TESTS BUT THIS IS NOT NECESSARY RIGHT NOW

# ----------------------------
# Step 1: Download instance and reference solution
# ----------------------------
instance_name = "X-n101-k25"  # change to any VRPLIB instance
instance_file = vrplib.download_instance(f"{instance_name}.vrp")
solution_file = vrplib.download_solution(f"{instance_name}.sol")

# ----------------------------
# Step 2: Load instance into our solver
# ----------------------------
cvrp_instance = CVRPInstance(instance_file)
cvrp_instance.summary()

# ----------------------------
# Step 3: Create and run Ant Colony solver
# ----------------------------
aco = AntColonyCVRP(
    dist_matrix=cvrp_instance.dist_matrix,
    demands=cvrp_instance.demands,
    vehicle_capacity=cvrp_instance.vehicle_capacity,
    num_ants=20,
    max_iter=100,
    alpha=1.0,
    beta=2.0,
    rho=0.5,
    Q=100,
    local_search=True,
    rng_seed=42
)

start_time = time.time()
best_routes, best_cost = aco.run(verbose=True)
elapsed_time = time.time() - start_time

# ----------------------------
# Step 4: Evaluate against reference solution
# ----------------------------
# Load reference solution using vrplib
optimal_sol = vrplib.read_solution(solution_file)

# Calculate gap
gap = 100 * (best_cost - optimal_sol["cost"]) / optimal_sol["cost"]

# ----------------------------
# Step 5: Display results
# ----------------------------
from utils import display_results  # optional: use your display function

display_results(instance_file, best_cost, elapsed_time)

print(f"\nOptimal reference cost: {optimal_sol['cost']}")
print(f"Your solution cost: {best_cost:.2f}")
print(f"Gap vs reference: {gap:.2f}%")
