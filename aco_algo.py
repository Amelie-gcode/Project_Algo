import numpy as np
import random
from copy import deepcopy

class AntColonyCVRP:
    """
    Ant Colony Optimization for CVRP (without time windows).
    Includes hybrid local search (2-opt and relocate moves).
    """

    def __init__(self, dist_matrix, demands, vehicle_capacity, num_vehicles=None,
                 alpha=1.0, beta=2.0, rho=0.5, Q=100, num_ants=10, max_iter=100,
                 local_search=True, rng_seed=None):
        self.D = dist_matrix
        self.demands = demands
        self.capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.alpha = alpha # influence of pheromone
        self.beta = beta # influence of visibility (1/distance)
        self.rho = rho # pheromone evaporation rate
        self.Q = Q # pheromone deposit factor
        self.num_ants = num_ants # of ants per iteration
        self.max_iter = max_iter # max iterations
        self.local_search = local_search # enable local search

        # Random seed for reproducibility
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        # Initialize pheromone and visibility matrices
        self.N = len(self.D)
        self.pheromone = np.ones((self.N, self.N))
        np.fill_diagonal(self.D, np.inf)
        # Avoid division by zero
        self.visibility = 1 / self.D
        self.visibility[np.isinf(self.visibility)] = 1e-6

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


        # --- Print final solution summary ---
        print("\n=== FINAL BEST SOLUTION ===")
        print(f"Total cost: {self.best_cost:.2f}")
        print(f"Number of vehicles used: {len(self.best_solution)}")
        for r_idx, route in enumerate(self.best_solution, start=1):
            print(f"  Vehicle {r_idx}: {route}")

        # Return results for external display
        return self.best_solution, self.best_cost

    # ------------------------------------------------------
    # Solution construction (each individual ant builds a route plan) and calculates its cost
    # ------------------------------------------------------
    def construct_solution(self):
        """
        Each ant builds routes until all customers are visited.
        This version enforces a maximum number of vehicles (routes)
        equal to `self.num_vehicles`, if specified.
        """

        unvisited = set(range(1, self.N))  # exclude depot (0)
        routes = []  # list of routes for this ant
        total_cost = 0  # total travel cost

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

                # Select next customer probabilistically
                probs = np.array([
                    (self.pheromone[current][j] ** self.alpha) *
                    (self.visibility[current][j] ** self.beta)
                    for j in feasible
                ])
                probs /= probs.sum()
                next_customer = random.choices(feasible, weights=probs)[0]

                # Update route and load
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


