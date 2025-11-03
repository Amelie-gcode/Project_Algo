import math
import random
from typing import List, Tuple, Optional

from local_search import two_opt, three_opt


class Ant:
    def __init__(self, num_customers: int):
        self.unserved = set(range(1, num_customers))  # nodes 1..n-1 are customers (0 is depot)
        self.routes: List[List[int]] = []
        self.total_length: float = math.inf


def construct_solution(
    dist,
    demands: List[int],
    capacity: int,
    tau,
    alpha: float,
    beta: float,
    rng: random.Random,
    apply_two_opt: bool,
    apply_three_opt: bool,
    use_acs: bool = True,
    q0: float = 0.1,
    xi: float = 0.1,
    tau0: float = 1e-3,
) -> Tuple[List[List[int]], float]:
    n = len(demands)
    unserved = set(range(1, n))
    routes: List[List[int]] = []
    total_len = 0.0

    while unserved:
        load = 0
        route = [0]
        current = 0
        while True:
            candidates = [j for j in unserved if load + demands[j] <= capacity]
            if not candidates:
                break
            # Transition rule (ACS or roulette)
            desirabilities = []
            for j in candidates:
                eta = 1.0 / (dist[current][j] + 1e-9)
                desirabilities.append((j, (tau[current][j] ** alpha) * (eta ** beta)))
            if use_acs and rng.random() < q0:
                # Exploitation: choose argmax
                j = max(desirabilities, key=lambda x: x[1])[0]
            else:
                denom = sum(w for _, w in desirabilities)
                if denom <= 0:
                    j = rng.choice(candidates)
                else:
                    r = rng.random()
                    acc = 0.0
                    j = desirabilities[-1][0]
                    for node, w in desirabilities:
                        acc += w / denom
                        if r <= acc:
                            j = node
                            break
            route.append(j)
            load += demands[j]
            unserved.remove(j)
            # Local pheromone update (ACS)
            if use_acs:
                tij = tau[current][j]
                tau[current][j] = (1 - xi) * tij + xi * tau0
                tau[j][current] = tau[current][j]
            current = j
        route.append(0)
        # Local search
        if apply_three_opt:
            route = three_opt(route, dist)
        elif apply_two_opt:
            route = two_opt(route, dist)
        routes.append(route)
        # Update length
        for i in range(len(route) - 1):
            total_len += dist[route[i]][route[i + 1]]

    return routes, total_len


def update_pheromones(
    tau,
    evaporation: float,
    ants_solutions: List[Tuple[List[List[int]], float]],
    Q: float = 1.0,
    max_depositants: int = 5,
    tau_min: float = 1e-12,
    tau_max: float = 1e6,
):
    n = len(tau)
    # Evaporation
    for i in range(n):
        for j in range(n):
            tau[i][j] *= (1.0 - evaporation)
            if tau[i][j] < tau_min:
                tau[i][j] = tau_min
    # Deposit from top-k ants
    ants_solutions.sort(key=lambda x: x[1])
    for routes, length in ants_solutions[:max_depositants]:
        if length <= 0:
            continue
        deposit = Q / length
        for route in routes:
            for a, b in zip(route[:-1], route[1:]):
                tau[a][b] += deposit
                tau[b][a] += deposit
                if tau[a][b] > tau_max:
                    tau[a][b] = tau_max
                    tau[b][a] = tau_max


def run_aco(
    dist,
    demands: List[int],
    capacity: int,
    iterations: int,
    num_ants: int,
    alpha: float,
    beta: float,
    evaporation: float,
    seed: Optional[int],
    use_two_opt: bool,
    use_three_opt: bool,
    use_acs: bool = True,
    q0: float = 0.1,
    xi: float = 0.1,
):
    n = len(demands)
    if n < 2:
        raise ValueError("Instance has fewer than 2 nodes (depot + at least 1 customer)" )
    rng = random.Random(seed)

    # Initialize pheromones
    # Heuristic: init tau = 1 / (n * avg_distance)
    avg_d = sum(dist[i][j] for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    tau0 = 1.0 / max(avg_d * n, 1e-6)
    tau = [[tau0 for _ in range(n)] for _ in range(n)]
    # Pheromone bounds (MMAS-style simple bounds)
    tau_min = 0.1 * tau0
    tau_max = 10.0 * tau0

    best_routes: List[List[int]] = []
    best_len = math.inf
    history = []

    for it in range(iterations):
        ants_solutions: List[Tuple[List[List[int]], float]] = []
        for _ in range(num_ants):
            routes, length = construct_solution(
                dist,
                demands,
                capacity,
                tau,
                alpha,
                beta,
                rng,
                apply_two_opt=use_two_opt,
                apply_three_opt=use_three_opt,
                use_acs=use_acs,
                q0=q0,
                xi=xi,
                tau0=tau0,
            )
            ants_solutions.append((routes, length))
            if length < best_len:
                best_len = length
                best_routes = routes

        update_pheromones(tau, evaporation, ants_solutions, tau_min=tau_min, tau_max=tau_max)
        history.append(best_len)

    return best_routes, best_len, tau, history


def run_aco_iter(
    dist,
    demands: List[int],
    capacity: int,
    iterations: int,
    num_ants: int,
    alpha: float,
    beta: float,
    evaporation: float,
    seed: Optional[int],
    use_two_opt: bool,
    use_three_opt: bool,
    use_acs: bool = True,
    q0: float = 0.1,
    xi: float = 0.1,
    stagnation_iter: int = 200,
    reinit_factor: float = 0.5,
):
    n = len(demands)
    if n < 2:
        raise ValueError("Instance has fewer than 2 nodes (depot + at least 1 customer)")
    rng = random.Random(seed)

    avg_d = sum(dist[i][j] for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    tau0 = 1.0 / max(avg_d * n, 1e-6)
    tau = [[tau0 for _ in range(n)] for _ in range(n)]
    tau_min = 0.1 * tau0
    tau_max = 10.0 * tau0

    best_routes: List[List[int]] = []
    best_len = math.inf
    history = []

    for it in range(iterations):
        ants_solutions: List[Tuple[List[List[int]], float]] = []
        for _ in range(num_ants):
            routes, length = construct_solution(
                dist,
                demands,
                capacity,
                tau,
                alpha,
                beta,
                rng,
                apply_two_opt=use_two_opt,
                apply_three_opt=use_three_opt,
                use_acs=use_acs,
                q0=q0,
                xi=xi,
                tau0=tau0,
            )
            ants_solutions.append((routes, length))
            if length < best_len:
                best_len = length
                best_routes = routes

        update_pheromones(tau, evaporation, ants_solutions, tau_min=tau_min, tau_max=tau_max)
        history.append(best_len)

        # Simple stagnation control: reinitialize towards tau0 if no improvement recently
        if (it + 1) % max(1, stagnation_iter) == 0 and (len(history) < 2 or abs(history[-1] - history[-stagnation_iter]) < 1e-9):
            for i in range(n):
                for j in range(n):
                    tau[i][j] = (1 - reinit_factor) * tau[i][j] + reinit_factor * tau0

        yield it + 1, best_routes, best_len, tau, history


