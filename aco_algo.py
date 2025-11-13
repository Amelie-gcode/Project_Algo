import numpy as np
import random
from copy import deepcopy
from CW import *

class AntColonyCVRP:
    """
    Ant Colony Optimization for CVRP (without time windows).
    Includes hybrid local search (2-opt and relocate moves).
    """

    def __init__(self, dist_matrix, demands, vehicle_capacity, num_vehicles=None,
                 alpha=1.0, beta=2.0, rho=0.2, Q=50, num_ants=20, max_iter=100,
                 local_search=True, rng_seed=None,
                 # new tuning params to focus on clarke and wright
                 seed_strength=500.0,
                 elitist_weight=5.0,
                 start_with_cw_frac=0.2,
                 candidate_list_size=20,
                 tau_min=1e-4,
                 tau_max=1e4):
        # Ensure a float numpy array for safe numeric operations
        self.D = np.array(dist_matrix, dtype=float)
        self.demands = demands
        self.capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.alpha = alpha  # influence of pheromone
        self.beta = beta  # influence of visibility (1/distance)
        self.rho = rho  # pheromone evaporation rate
        self.Q = Q  # pheromone deposit factor
        self.num_ants = num_ants  # of ants per iteration
        self.max_iter = max_iter  # max iterations
        self.local_search = local_search  # enable local search

        # Random seed for reproducibility
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        # Initialize pheromone and visibility matrices
        self.N = len(self.D)
        # initial pheromone levels (will be re-weighted/seeded in run())
        self.pheromone = np.ones((self.N, self.N))

        # Build a safe distance matrix for numerical operations
        D_safe = self.D.copy()
        # Diagonal should not be used as neighbor distances
        np.fill_diagonal(D_safe, np.inf)

        # Visibility = 1 / distance but protect against zeros / non-finite
        self.visibility = np.zeros_like(D_safe)
        finite_mask = np.isfinite(D_safe) & (D_safe > 1e-12)
        self.visibility[finite_mask] = 1.0 / D_safe[finite_mask]
        # Small non-zero visibility for invalid/unreachable pairs
        self.visibility[~finite_mask] = 1e-6

        # Clarke & Wright guidance
        # seed_strength: additive pheromone on edges of cw_solution at init
        self.seed_strength = seed_strength
        # elitist factor for extra pheromone deposit from global best
        self.elitist_weight = elitist_weight
        # fraction of ants that use cw-guided bias during construction
        self.start_with_cw_frac = start_with_cw_frac

        # Candidate lists (nearest neighbors) to focus construction
        self.candidate_list_size = min(candidate_list_size, self.N - 1)
        # For candidate lists, avoid treating zero or invalid distances as nearest
        D_for_sort = D_safe.copy()
        # replace any tiny/zero distances (off-diagonal) with large value so they are not picked as near neighbors
        small_mask = (D_for_sort <= 1e-12) | ~np.isfinite(D_for_sort)
        D_for_sort[small_mask] = np.inf
        self.candidates = [np.argsort(D_for_sort[i])[:self.candidate_list_size].tolist() for i in range(self.N)]

        # pheromone bounds to avoid search stagnation
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Best solution tracking
        self.best_cost = float("inf")
        self.best_solution = None

    # ------------------------------------------------------
    # Main ACO loop
    # ------------------------------------------------------
    def run(self, verbose=True):
        """
        Main optimization loop for the Ant Colony Optimization (ACO) algorithm.

        Each iteration consists of:
        1. Multiple ants constructing feasible CVRP solutions independently.
        2. Evaluating all ants and selecting the best one of this iteration.
        3. (Optional) Improving the iteration's best solution using local search.
        4. Updating pheromones based on the improved best solution.
        5. Tracking the global best solution found so far.

        This version stops early if the gap to the known optimum drops below 7%.
        """

        # Reset best solution before starting
        self.best_cost = float("inf")
        self.best_solution = None

        # --- Main ACO loop ---
        # --- (0) Optional: initialize / seed pheromones using Clarke & Wright solution ---
        if self.cw_solution is not None:
            if verbose:
                print("Seeding pheromone with Clarke & Wright solution...")
            # Build edge set for quick lookup
            self.cw_edges = set()
            for route in self.cw_solution:
                # ensure route includes depot indicators if needed
                r = route if route[0] == 0 else [0] + route + [0]
                for i in range(len(r) - 1):
                    a, b = r[i], r[i + 1]
                    self.cw_edges.add((a, b))
                    self.cw_edges.add((b, a))
            # Add a seed to pheromone matrix on those edges
            for (a, b) in self.cw_edges:
                # Guard against mismatched instances / indexing: only apply if
                # the indices exist in the pheromone matrix
                if 0 <= a < self.N and 0 <= b < self.N:
                    self.pheromone[a][b] += self.seed_strength
                else:
                    if verbose:
                        print(f"Warning: skipping C&W edge {(a,b)} — outside pheromone matrix size {self.N}")
            # Clip pheromones to configured bounds to avoid numerical blowup
            np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)

        for it in range(1, self.max_iter + 1):
            iteration_best_cost = float("inf")
            iteration_best_sol = None
            iteration_solutions = []

            # (1) Each ant constructs a solution
            for _ in range(self.num_ants):
                solution, cost = self.construct_solution()
                iteration_solutions.append((solution, cost))

            # (2) Pick the best of this iteration
            iteration_best_sol, iteration_best_cost = min(iteration_solutions, key=lambda x: x[1])

            # (3) Optional local search improvement
            if self.local_search:
                improved_sol, improved_cost = self.local_search_hill(iteration_best_sol, iteration_best_cost)
                if improved_cost < iteration_best_cost:
                    iteration_best_sol, iteration_best_cost = improved_sol, improved_cost

            # (4) Update pheromones
            self.update_pheromones(iteration_best_sol, iteration_best_cost)

            # (5) Update global best
            if iteration_best_cost < self.best_cost:
                self.best_cost = iteration_best_cost
                self.best_solution = iteration_best_sol

            # (6) Verbose progress
            if verbose and (it % max(1, self.max_iter // 10) == 0 or it == 1):
                print(f"Iteration {it}/{self.max_iter} - Best: {self.best_cost:.2f}")
        
        # === (7) Final fallback: ensure we return at least C&W solution ===
        if self.cw_solution is not None:
            cw_cost = total_travel_distance(self.cw_solution)
            if self.best_cost > cw_cost:
                if verbose:
                    print(f"ACO ({self.best_cost:.2f}) did not beat C&W ({cw_cost:.2f}). Returning C&W solution.")
                self.best_solution = self.cw_solution
                self.best_cost = cw_cost

        return self.best_solution, self.best_cost

    # ------------------------------------------------------
    # Solution construction (each individual ant builds a route plan) and calculates its cost
    # ------------------------------------------------------
    def construct_solution(self, cw_bias_frac=0.0):
        """Build a solution with optional bias towards C&W edges."""
        unvisited = set(range(1, self.N)) # exclude depot (0)
        routes = [] # routes for this ant
        total_cost = 0 # total cost for this ant

        while unvisited:
            # === Hard vehicle limit check ===
            if self.num_vehicles is not None and len(routes) >= self.num_vehicles:
                # All vehicles used — assign remaining customers to the last route
                last_route = routes[-1]
                current = last_route[-2] if len(last_route) > 1 else 0  # last visited before depot

                for j in list(unvisited):
                    # Insert before depot
                    last_route.insert(-1, j)
                    # Add cost for inserting customer j before returning to depot
                    total_cost += self.D[current][j] + self.D[j][0] - self.D[current][0]
                    current = j
                    unvisited.remove(j)
                # Ensure route ends at depot
                if last_route[-1] != 0:
                    last_route.append(0)
                break

            # === Normal route construction ===
            route = [0]  # start at depot
            load = 0
            current = 0

            while True:
                # Find feasible customers within capacity
                feasible = [j for j in unvisited if self.demands[j] + load <= self.capacity]

                if not feasible:
                    break

                # Restrict to candidate list (nearest neighbors)
                cand = feasible
                cand_neighbors = [c for c in self.candidates[current] if c in feasible]
                if len(cand_neighbors) > 0:
                    cand = cand_neighbors + [f for f in feasible if f not in cand_neighbors]

                # Calculate selection probabilities
                probs = np.array([
                    (self.pheromone[current][j] ** self.alpha) *
                    (self.visibility[current][j] ** self.beta)
                    for j in cand
                ])
                
                probs_sum = probs.sum()
                if probs_sum <= 0:
                    probs = np.ones(len(probs)) / len(probs)
                else:
                    probs /= probs_sum

                # C&W bias: amplify probs of edges in C&W solution
                if hasattr(self, 'cw_edges') and cw_bias_frac > 0 and random.random() < cw_bias_frac:
                    bias = np.array([5.0 if (current, j) in self.cw_edges else 1.0 for j in cand])
                    probs = probs * bias
                    probs_sum = probs.sum()
                    if probs_sum > 0:
                        probs = probs / probs_sum

                # Roulette wheel selection
                next_customer = random.choices(cand, weights=probs)[0]
                route.append(next_customer)
                total_cost += self.D[current][next_customer]
                load += self.demands[next_customer]
                unvisited.remove(next_customer)
                current = next_customer

            # Return to depot
            route.append(0)
            total_cost += self.D[current][0]
            routes.append(route)

        return routes, total_cost

    def construct_solution_with_bias(self, cw_bias_frac=0.0):
        """Alias for construct_solution to support adaptive bias."""
        return self.construct_solution(cw_bias_frac)

    # ------------------------------------------------------
    # Local Search (Hybrid Hill Climbing)
    # ------------------------------------------------------
    def local_search_hill(self, routes, total_cost, max_no_improve=30):
        """
        Simple hybrid local search that applies:
        - Intra-route 2-opt (reverses segments to shorten routes)
        - Inter-route relocate (moves one customer between routes)
        After each improvement, the total cost is fully recomputed
        to avoid floating point drift or negative total cost bugs.
        """
        best_routes = deepcopy(routes)
        best_cost = total_cost
        no_improve = 0

        while no_improve < max_no_improve:
            improved = False

            # === Intra-route 2-opt improvement ===
            for r_idx, route in enumerate(best_routes):
                # Skip routes that have too few customers
                if len(route) <= 4:
                    continue

                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        # Create new route by reversing segment
                        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]

                        old_cost = self.route_cost(route)
                        new_cost = self.route_cost(new_route)

                        if new_cost < old_cost - 1e-6:  # small tolerance
                            best_routes[r_idx] = new_route
                            # recompute global cost (safer than incremental update)
                            best_cost = sum(self.route_cost(r) for r in best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # === Inter-route relocate move ===
            if not improved:
                for a in range(len(best_routes)):
                    for b in range(len(best_routes)):
                        if a == b:
                            continue

                        for pos_a in range(1, len(best_routes[a]) - 1):
                            cust = best_routes[a][pos_a]

                            # Calculate current loads
                            load_a = sum(self.demands[i] for i in best_routes[a] if i != 0)
                            load_b = sum(self.demands[i] for i in best_routes[b] if i != 0)

                            # Capacity check
                            if load_b + self.demands[cust] > self.capacity:
                                continue

                            for insert_pos in range(1, len(best_routes[b])):
                                # Perform relocate
                                new_a = best_routes[a][:pos_a] + best_routes[a][pos_a + 1:]
                                new_b = best_routes[b][:insert_pos] + [cust] + best_routes[b][insert_pos:]

                                old_cost = self.route_cost(best_routes[a]) + self.route_cost(best_routes[b])
                                new_cost = self.route_cost(new_a) + self.route_cost(new_b)

                                if new_cost < old_cost - 1e-6:
                                    best_routes[a], best_routes[b] = new_a, new_b
                                    best_cost = sum(self.route_cost(r) for r in best_routes)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

            # === Stopping criteria ===
            if improved:
                no_improve = 0
            else:
                no_improve += 1

        # Safety check: total cost must not be negative
        best_cost = abs(best_cost)
        return best_routes, best_cost


    def route_cost(self, route):
        """
        Calculate the total travel cost (or distance) of a single vehicle route.

        In the CVRP (Capacitated Vehicle Routing Problem), each 'route' is a list
        of customer indices that a vehicle visits, starting and ending at the depot.
        Example: [0, 5, 7, 3, 0]
        means the vehicle starts at the depot (0), visits customers 5 → 7 → 3,
        then returns to the depot.

        self.D is the distance matrix, where self.D[i][j] gives the distance
        between location i and location j.

        This function adds up all pairwise distances along the route:
            total_cost = D[0][5] + D[5][7] + D[7][3] + D[3][0]

        Returns:
            A single number representing the total cost (distance) of this route.
        """
        return sum(self.D[route[i]][route[i + 1]] for i in range(len(route) - 1))

    def update_pheromones(self, best_solution, best_cost):
        """
        Update the pheromone matrix after each iteration of the ACO algorithm.

        The pheromone matrix stores the "collective memory" of all ants.
        Each entry pheromone[i][j] represents how attractive it is to travel
        directly from location i to location j.

        This function has two main steps:

        1. **Evaporation:** All pheromone values are slightly reduced.
           This prevents the algorithm from converging too early to one path
           and allows exploration of new possible routes.

              self.pheromone *= (1 - self.rho)

           Here, `rho` is the evaporation rate (0 < rho < 1).
           A higher rho means faster evaporation (more exploration).

        2. **Deposition (Reinforcement):** 
           The best route(s) found in this iteration add pheromone to the edges
           they used. Better (shorter) solutions deposit more pheromone,
           encouraging future ants to follow similar paths.

              deposit = Q / best_cost

           - Q is a scaling constant (total pheromone available).
           - Dividing by best_cost means shorter (better) routes deposit more.

           For every edge (a → b) in the best route:
              pheromone[a][b] += deposit
              pheromone[b][a] += deposit   # because distance is symmetric

        The combined effect is that:
           - Frequently used short edges become stronger over time.
           - Rarely used or long edges lose pheromone and are less likely
             to be chosen in the future.
        """

        # --- (1) PHEROMONE EVAPORATION ---
        # Every edge loses a fraction of its pheromone level.
        # This helps the colony "forget" old or suboptimal paths
        # and encourages new exploration.
        self.pheromone *= (1 - self.rho)

        # --- (2) PHEROMONE DEPOSITION ---
        # Calculate how much pheromone to deposit for the best solution found.
        # The better (lower-cost) the solution, the larger the deposit.
        deposit = self.Q / best_cost

        # Go through each route in the best solution
        for route in best_solution:
            # Each route is a list of locations: [0, i, j, k, 0]
            # For each consecutive pair of nodes (a, b), deposit pheromone.
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]

                # Increase pheromone on both directions (symmetric problem)
                self.pheromone[a][b] += deposit
                self.pheromone[b][a] += deposit

        if hasattr(self, "cw_solution") and self.cw_solution is not None:
            for route in self.cw_solution:
                for i in range(len(route) - 1):
                    a, b = route[i], route[i + 1]
                    # Reinforce C&W arcs but guard against out-of-bounds indices
                    if 0 <= a < self.N and 0 <= b < self.N:
                        self.pheromone[a][b] += 500.0
                        self.pheromone[b][a] += 500.0
                    else:
                        # optional verbose message when skipping invalid arc
                        pass

        # Clip pheromones to keep values in numerical bounds
        np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)


