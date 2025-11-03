from typing import List


def route_length(route: List[int], dist) -> float:
    total = 0.0
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    return total


def two_opt(route: List[int], dist) -> List[int]:
    improved = True
    best = route
    best_len = route_length(best, dist)
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                delta = (
                    dist[best[i - 1]][best[k]]
                    + dist[best[i]][best[k + 1]]
                    - dist[best[i - 1]][best[i]]
                    - dist[best[k]][best[k + 1]]
                )
                if delta < -1e-9:
                    new_route = best[:i] + list(reversed(best[i : k + 1])) + best[k + 1 :]
                    best = new_route
                    best_len += delta
                    improved = True
                    break
            if improved:
                break
    return best


def three_opt(route: List[int], dist) -> List[int]:
    # Basic 3-opt (evaluate a subset of reconnections for speed)
    n = len(route)
    best = route
    best_len = route_length(best, dist)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 4):
            for j in range(i + 1, n - 2):
                for k in range(j + 1, n - 1):
                    a, b, c, d, e, f = best[i - 1], best[i], best[j - 1], best[j], best[k - 1], best[k]
                    base = dist[a][b] + dist[c][d] + dist[e][f]
                    # Candidate reconnections (subset)
                    candidates = []
                    # Case 1: reverse (b..c)
                    cand1 = best[:i] + list(reversed(best[i:j])) + best[j:]
                    cand1_len = best_len - base + dist[a][best[j - 1]] + dist[b][d] + dist[e][f]
                    candidates.append((cand1_len, cand1))
                    # Case 2: reverse (d..e)
                    cand2 = best[:j] + list(reversed(best[j:k])) + best[k:]
                    cand2_len = best_len - base + dist[a][b] + dist[c][best[k - 1]] + dist[d][f]
                    candidates.append((cand2_len, cand2))
                    # Case 3: reverse (b..e)
                    cand3 = best[:i] + list(reversed(best[i:k])) + best[k:]
                    cand3_len = best_len - base + dist[a][best[k - 1]] + dist[c][d] + dist[b][f]
                    candidates.append((cand3_len, cand3))

                    for cand_len, cand in candidates:
                        if cand_len + 1e-9 < best_len:
                            best = cand
                            best_len = cand_len
                            n = len(best)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return best


