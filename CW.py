"""
CW.py
Clarke & Wright initializer and simple VRP utilities.

This module provides a small Clarke & Wright savings heuristic implementation
and a distance/solution evaluator. English comments explain the purpose of
each section and the algorithmic steps for clarity.
"""

from functions import get_vrptw_instance
from functions import show
import numpy as np
from scipy.spatial import distance_matrix

# ---------------------------------------------------------------------------
# Instance loading
# - Define the instance file path and load data using helper `get_vrptw_instance`.
# - `data_set` contains the problem (coordinates, demands, capacity and
#   possibly an `edge_weight` matrix). `data_set_solution` may contain a
#   reference (best-known) solution used for comparison.
# ---------------------------------------------------------------------------
INSTANCE_PATH = r"C:\Users\ameli\Documents\A3\Algo_combi_opti\Project_Algo\instances\M\M-n101-k10.vrp"
data_set, data_set_solution = get_vrptw_instance(INSTANCE_PATH)
BEST_SOLUTION = data_set_solution['routes']

# Problem parameters extracted from the dataset
capacity = data_set['capacity']
coords = data_set['node_coord']
demand = data_set['demand']
# Number of clients excluding depot (index 0)
NB_CLIENT = len(coords) - 1

"""
Simple Clarke & Wright implementation and utility helpers for the VRP instance.

This module loads a VRP instance, computes distance/edge-weight matrices,
provides a total_travel_distance evaluator and a basic Clarke & Wright
(`clark_and_wright`) initializer. Comments in the file explain each step
to help future readers understand the algorithm and the data handling.
"""


# ---------------------------------------------------------------------------
# Distance matrices
# - Compute a Euclidean distance matrix from coordinates as a fallback.
# - Prefer using the instance `edge_weight` matrix when available because it
#   reflects the instance's official costs (VRPLIB metric). If absent, use
#   the Euclidean DISTANCE computed from coords.
# ---------------------------------------------------------------------------
points = np.array(coords)
DISTANCE = distance_matrix(points, points)

# Prefer instance-provided edge weights if present
EDGE_WEIGHT = data_set.get('edge_weight', None)
if EDGE_WEIGHT is None:
    EDGE_WEIGHT = DISTANCE

def total_travel_distance(solution):
        """Evaluate the total cost/distance of a VRP solution.

        Parameters
        - solution: list of routes, where each route is a list of customer indices
            (example route: [5, 2, 8]). The depot is assumed to be index 0.

        Returns
        - total distance (float) computed using the instance edge weight matrix
            if available, otherwise the Euclidean distance matrix.
        """
        dist_tot = 0.0
        for route in solution:
                # Add depot at start and end to compute full route cost
                full_route = [0] + route + [0]
                for client in range(len(full_route) - 1):
                        origin = full_route[client]
                        destination = full_route[client + 1]
                        dist_tot += EDGE_WEIGHT[origin, destination]

        return float(dist_tot)

def clark_and_wright():
    """Build an initial solution using the Clarke & Wright savings heuristic.

    The algorithm constructs savings S(i,j) and iteratively merges routes
    in order of decreasing savings, ensuring each merged route respects the
    vehicle capacity. Only route-end to route-end merges are allowed to keep
    concatenation straightforward.
    """
    # Use instance edge weights if available for accurate cost computation
    distance = data_set['edge_weight']

    # Start with each customer in their own route (no depot in stored route)
    solution = [np.array([i]) for i in range(1, NB_CLIENT + 1)]

    # Compute savings: Sij = c(i,0) + c(0,j) - c(i,j)
    dist = np.array(distance, dtype=float).copy()
    if dist.shape[0] == dist.shape[1]:
        np.fill_diagonal(dist, 0.0)
    economic_matrix = dist[:, 0][:, np.newaxis] + dist[0, :] - dist

    # Consider only i < j (upper triangle) and sort savings descending
    i_indice, j_indice = np.triu_indices_from(economic_matrix, k=1)
    values = economic_matrix[i_indice, j_indice]
    arg_sort = np.argsort(values)[::-1]

    sorted_Sij = values[arg_sort]
    i_sorted_idx = i_indice[arg_sort]
    j_sorted_idx = j_indice[arg_sort]

    # Iterate over sorted savings and attempt merges when feasible
    for k in range(len(sorted_Sij)):
        client_i = i_sorted_idx[k]
        client_j = j_sorted_idx[k]

        route_client_i = None
        route_client_j = None
        idx_i = None
        idx_j = None

        # Locate the current routes containing the two clients
        for idx, route in enumerate(solution):
            if client_i in route:
                route_client_i = route
                idx_i = idx
            if client_j in route:
                route_client_j = route
                idx_j = idx

        # Skip if clients are in the same route or not found
        if idx_i is None or idx_j is None or idx_i == idx_j:
            continue

        # Check whether the clients are at route ends (only then concatenation is simple)
        is_i_start = (client_i == route_client_i[0])
        is_i_end = (client_i == route_client_i[-1])
        is_j_start = (client_j == route_client_j[0])
        is_j_end = (client_j == route_client_j[-1])

        if not ((is_i_start or is_i_end) or (is_j_start or is_j_end)):
            continue

        new_route = None

        # Try the four concatenation patterns (preserve visit order)
        if is_i_end and is_j_start:
            potential_new_route = np.concatenate([route_client_i, route_client_j])
            if is_route_feasable(potential_new_route):
                new_route = potential_new_route

        elif is_j_end and is_i_start:
            potential_new_route = np.concatenate([route_client_j, route_client_i])
            if is_route_feasable(potential_new_route):
                new_route = potential_new_route

        elif is_i_start and is_j_start:
            potential_new_route = np.concatenate([route_client_i[::-1], route_client_j])
            if is_route_feasable(potential_new_route):
                new_route = potential_new_route

        elif is_i_end and is_j_end:
            potential_new_route = np.concatenate([route_client_i, route_client_j[::-1]])
            if is_route_feasable(potential_new_route):
                new_route = potential_new_route

        # If merge succeeded, replace the two routes by the merged one
        if new_route is not None:
            if idx_j > idx_i:
                solution[idx_i] = new_route
                del solution[idx_j]
            else:
                solution[idx_j] = new_route
                del solution[idx_i]

    # Return Python lists (not numpy arrays)
    return [route.tolist() for route in solution]

def is_route_feasable(route):
    """Check whether the provided route respects vehicle capacity.

    The input `route` is a sequence/array of customer indices (depot not
    included). The function sums the customers' demands and returns True if
    the total demand fits into the vehicle capacity.
    """
    if sum(demand[k] for k in route) <= capacity:
        return True


print('===== Clarke and wright ======')
sol = clark_and_wright()
print(total_travel_distance(sol))
BEST_SOLUTION_VALUE= total_travel_distance(BEST_SOLUTION)
gap =  100 * (total_travel_distance(sol)-BEST_SOLUTION_VALUE) / BEST_SOLUTION_VALUE 
print ("gap cw :", gap)
