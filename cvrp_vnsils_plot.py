# ================================================================
# CVRP - VNS-ILS + affichage façon "captures d'écran"
# - Charge les instances VRPLIB, distances EUC_2D arrondies
# - Résout avec le VNS-ILS du code fourni
# - Affiche : instance (scatter) + solution (routes colorées)
# ================================================================

import os
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

import vrplib

# ----------------------------
# Distances EUC_2D arrondies
# ----------------------------
def euc2d_rounded(coords: np.ndarray) -> np.ndarray:
    raw = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return np.rint(raw).astype(int)

# ----------------------------
# Chargement instance
# ----------------------------
BKS = {
    "A-n32-k5.vrp": 784,
    "X-n101-k25.vrp": 27591,
}

def load_instance(name: str):
    """Charge une instance (local ./data/<name> ou ./<name>, sinon téléchargement).
       Remplace la matrice de distances par EUC_2D arrondies."""
    local_paths = [os.path.join("data", name), name]
    inst = None
    for p in local_paths:
        if os.path.exists(p):
            inst = vrplib.read_instance(p)
            break
    if inst is None:
        inst = vrplib.read_instance(name)

    coords = np.asarray(inst["node_coord"], dtype=float)
    inst["edge_weight"] = euc2d_rounded(coords)
    inst["capacity"]     = int(inst.get("capacity", inst.get("CAPACITY", 999999)))
    inst["vehicles"]     = int(inst.get("vehicles", inst.get("VEHICLES", 999999)))
    inst["demand"]       = np.asarray(inst["demand"], dtype=int)
    depot                = inst.get("depot", np.array([0], dtype=int))
    inst["depot"]        = np.asarray(depot, dtype=int)
    assert inst["depot"][0] == 0, "Dépôt attendu à l'index 0 (0-based)."
    return inst

# ----------------------------
# Solveur VNS-ILS (identique)
# ----------------------------
class CVRPSolverVNSILS:
    def __init__(self, inst, seed=None):
        self.inst = inst
        self.n = len(inst["demand"])
        self.depot = int(inst["depot"][0])
        self.Q = int(inst["capacity"])
        self.K = max(1, int(inst.get("vehicles", 999999)))
        self.demand = inst["demand"]
        self.dist = inst["edge_weight"]
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def route_cost(self, r): return int(sum(self.dist[a, b] for a, b in zip(r[:-1], r[1:])))
    def cost(self, routes): return int(sum(self.route_cost(r) for r in routes))
    def route_load(self, r): return int(sum(self.demand[i] for i in r[1:-1]))

    def is_route_ok(self, r):
        if r[0] != self.depot or r[-1] != self.depot: return False
        mid = r[1:-1]
        if any(v == self.depot for v in mid): return False
        if len(mid) != len(set(mid)): return False
        if self.route_load(r) > self.Q: return False
        return True

    def is_solution_valid(self, sol):
        seen = []
        for r in sol:
            if not self.is_route_ok(r): return False
            seen += r[1:-1]
        return set(seen) == set(range(1, self.n)) and len(seen) == self.n - 1

    def initial_solution(self):
        nodes = list(range(1, self.n))
        random.shuffle(nodes)
        routes = []
        nb_try = self.K if self.K < 10**6 else max(1, (self.n - 1) // 8)
        for _ in range(nb_try):
            load = 0
            r = [self.depot]
            i = 0
            while i < len(nodes):
                v = nodes[i]
                if load + self.demand[v] <= self.Q:
                    r.append(v); load += self.demand[v]; nodes.pop(i)
                else:
                    i += 1
            if len(r) > 1:
                r.append(self.depot)
                if self.is_route_ok(r):
                    routes.append(r)
        for v in nodes:
            routes.append([self.depot, v, self.depot])
        assert self.is_solution_valid(routes)
        return routes

    # --- voisinages ---
    def two_opt_intra(self, routes):
        improved = False
        best = [r[:] for r in routes]
        best_c = self.cost(best)
        for idx, r in enumerate(routes):
            if len(r) <= 3: continue
            cur = r[:]
            cur_c = self.route_cost(cur)
            changed = True
            while changed:
                changed = False
                for i in range(1, len(cur)-2):
                    for j in range(i+1, len(cur)-1):
                        cand = cur[:i] + cur[i:j+1][::-1] + cur[j+1:]
                        if not self.is_route_ok(cand): continue
                        c = self.route_cost(cand)
                        if c < cur_c:
                            cur, cur_c = cand, c
                            changed = True
            if cur_c < self.route_cost(r):
                tmp = [rr[:] for rr in routes]
                tmp[idx] = cur
                c_all = self.cost(tmp)
                if c_all < best_c:
                    best, best_c, improved = tmp, c_all, True
        return improved, best

    def relocate(self, routes):
        improved = False
        best = [r[:] for r in routes]
        best_c = self.cost(best)
        for i in range(len(routes)):
            ri = routes[i]
            if len(ri) <= 3: continue
            for pi in range(1, len(ri)-1):
                node = ri[pi]
                for j in range(len(routes)):
                    for pj in range(1, len(routes[j])):
                        if i == j and (pj == pi or pj == pi+1): continue
                        new_routes = [r[:] for r in routes]
                        new_routes[i] = new_routes[i][:pi] + new_routes[i][pi+1:]
                        if len(new_routes[i]) == 2:
                            new_routes.pop(i)
                            j_adj = j - 1 if j > i else j
                        else:
                            j_adj = j
                        if j_adj < 0 or j_adj >= len(new_routes): continue
                        cand = new_routes[j_adj][:pj] + [node] + new_routes[j_adj][pj:]
                        if not self.is_route_ok(cand): continue
                        new_routes[j_adj] = cand
                        if not self.is_solution_valid(new_routes): continue
                        c_all = self.cost(new_routes)
                        if c_all < best_c:
                            best, best_c, improved = new_routes, c_all, True
        return improved, best

    def swap_1_1(self, routes):
        improved = False
        best = [r[:] for r in routes]
        best_c = self.cost(best)
        m = len(routes)
        for i in range(m):
            for j in range(i+1, m):
                ri, rj = routes[i], routes[j]
                if len(ri) <= 3 or len(rj) <= 3: continue
                for pi in range(1, len(ri)-1):
                    for pj in range(1, len(rj)-1):
                        a, b = ri[pi], rj[pj]
                        ni = ri[:pi] + [b] + ri[pi+1:]
                        nj = rj[:pj] + [a] + rj[pj+1:]
                        if not self.is_route_ok(ni) or not self.is_route_ok(nj): continue
                        tmp = [r[:] for r in routes]; tmp[i], tmp[j] = ni, nj
                        if not self.is_solution_valid(tmp): continue
                        c_all = self.cost(tmp)
                        if c_all < best_c:
                            best, best_c, improved = tmp, c_all, True
        return improved, best

    def cross_exchange(self, routes, max_len=2):
        improved = False
        best = [r[:] for r in routes]
        best_c = self.cost(best)
        m = len(routes)
        for i in range(m):
            for j in range(i+1, m):
                ri, rj = routes[i], routes[j]
                for li in range(1, min(max_len, len(ri)-2)+1):
                    for lj in range(1, min(max_len, len(rj)-2)+1):
                        for si in range(1, len(ri)-li):
                            for sj in range(1, len(rj)-lj):
                                segi = ri[si:si+li]; segj = rj[sj:sj+lj]
                                ni = ri[:si] + segj + ri[si+li:]
                                nj = rj[:sj] + segi + rj[sj+lj:]
                                if not self.is_route_ok(ni) or not self.is_route_ok(nj): continue
                                tmp = [r[:] for r in routes]; tmp[i], tmp[j] = ni, nj
                                if not self.is_solution_valid(tmp): continue
                                c_all = self.cost(tmp)
                                if c_all < best_c:
                                    best, best_c, improved = tmp, c_all, True
        return improved, best

    def vnd(self, routes):
        neighs = [self.two_opt_intra, self.relocate, self.swap_1_1, self.cross_exchange]
        k = 0
        cur = [r[:] for r in routes]
        while k < len(neighs):
            improved, new = neighs[k](cur)
            if improved:
                cur = new; k = 0
            else:
                k += 1
        return cur

    def shake(self, routes, strength=2):
        cur = [r[:] for r in routes]
        for _ in range(strength):
            if random.random() < 0.5 and len(cur) > 0:
                all_pos = [(ri, pi) for ri, r in enumerate(cur) for pi in range(1, len(r)-1)]
                if not all_pos: continue
                ri, pi = random.choice(all_pos); node = cur[ri][pi]
                cur[ri] = cur[ri][:pi] + cur[ri][pi+1:]
                if len(cur[ri]) == 2: cur.pop(ri)
                if len(cur) == 0:
                    cur = [[self.depot, node, self.depot]]
                else:
                    rj = random.randrange(len(cur))
                    pos = random.randrange(1, len(cur[rj]))
                    cand = cur[rj][:pos] + [node] + cur[rj][pos:]
                    if self.is_route_ok(cand): cur[rj] = cand
                    else: cur.append([self.depot, node, self.depot])
            else:
                if len(cur) < 2: continue
                i, j = random.sample(range(len(cur)), 2)
                if len(cur[i]) <= 3 or len(cur[j]) <= 3: continue
                pi = random.randrange(1, len(cur[i])-1)
                pj = random.randrange(1, len(cur[j])-1)
                a, b = cur[i][pi], cur[j][pj]
                ni = cur[i][:pi] + [b] + cur[i][pi+1:]
                nj = cur[j][:pj] + [a] + cur[j][pj+1:]
                if self.is_route_ok(ni) and self.is_route_ok(nj):
                    cur[i], cur[j] = ni, nj
        assert self.is_solution_valid(cur)
        return cur

    def solve(self, time_limit=10.0, restarts=4, ils_iters=200, shake_min=1, shake_max=4):
        best_sol, best_cost = None, math.inf
        t_end = time.time() + time_limit
        for _ in range(restarts):
            cur = self.initial_solution()
            cur = self.vnd(cur)
            cur_c = self.cost(cur)
            k = shake_min
            it = 0
            while time.time() < t_end and it < ils_iters:
                it += 1
                shaken = self.shake(cur, strength=k)
                shaken = self.vnd(shaken)
                s_c = self.cost(shaken)
                if s_c < cur_c:
                    cur, cur_c = shaken, s_c
                    k = shake_min
                else:
                    k = min(k+1, shake_max)
            if cur_c < best_cost:
                best_sol, best_cost = [r[:] for r in cur], cur_c
        return best_sol, best_cost

# ----------------------------
# Métriques & impression
# ----------------------------
def star_baseline(inst):
    depot = int(inst["depot"][0])
    return int(sum(2 * inst["edge_weight"][depot, i]
                   for i in range(1, len(inst["demand"]))
                   if inst["demand"][i] > 0))

def print_solution(inst, routes, cost, bks, total_time, title):
    gap = 100.0 * (cost - bks) / bks
    co2 = (star_baseline(inst) - cost) * 0.2
    print(f"\n==================== RESULTS - {title} ====================")
    print(f"BKS (optimum)        : {bks}")
    print(f"Star baseline        : {star_baseline(inst)}")
    print(f"Best cost            : {cost}")
    print(f"GAP vs BKS           : {gap:+.2f}%")
    print(f"CO₂ saved (proxy)    : {co2:.1f} kg")
    print(f"Total time           : {total_time:.2f} s")
    print("========================================================")
    print("\nBest routes:")
    for k, r in enumerate(routes, 1):
        load = int(sum(inst["demand"][i] for i in r[1:-1]))
        rc = int(sum(inst["edge_weight"][a, b] for a, b in zip(r[:-1], r[1:])))
        print(f"  V{k:02d}  load={load:3d}  cost={rc:4d}  route={r}")

# ----------------------------
# Fonctions d'affichage
# ----------------------------
def plot_instance(inst, title="Instance", savepath=None):
    coords = np.asarray(inst["node_coord"], dtype=float)
    depot  = int(inst["depot"][0])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # clients
    cust_idx = [i for i in range(len(coords)) if i != depot]
    ax.scatter(coords[cust_idx, 0], coords[cust_idx, 1], s=35, label="customers")
    # dépôt
    ax.scatter(coords[depot, 0], coords[depot, 1], s=60, marker='o', edgecolor='k', label="depot")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches='tight')
    return fig, ax

def plot_solution(inst, routes, title="Solution", savepath=None):
    coords = np.asarray(inst["node_coord"], dtype=float)
    depot  = int(inst["depot"][0])

    colors = ["tab:blue","tab:orange","tab:olive","tab:green","tab:purple",
              "tab:red","tab:pink","tab:brown","tab:cyan","tab:gray"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)

    # tracer chaque route
    for k, r in enumerate(routes, 1):
        c = colors[(k-1) % len(colors)]
        xs = [coords[i,0] for i in r]
        ys = [coords[i,1] for i in r]
        # trait plein pour la tournée
        ax.plot(xs, ys, marker='o', linewidth=2, label=f"Route {k}", color=c)
        # retour en tirets vers le dépôt (optionnel pour rappeler la structure)
        ax.plot([coords[r[-2],0], coords[depot,0]],
                [coords[r[-2],1], coords[depot,1]],
                linestyle='--', linewidth=1.5, color=c, alpha=0.8)

    # dépôt en surbrillance
    ax.scatter(coords[depot,0], coords[depot,1], s=70, edgecolor='k', zorder=5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches='tight')
    return fig, ax

# ----------------------------
# Run + affichage (comme les screenshots)
# ----------------------------
def run_and_plot(filename: str, runs=5, time_limit=5.0, seed=1234,
                 vns_restarts=4, vns_ils_iters=200, vns_shake_min=1, vns_shake_max=4):
    inst = load_instance(filename)
    bks = BKS.get(filename, 0)

    # 1) Affiche l'instance
#    plot_instance(inst, title=f"{filename} (n={len(inst['demand'])-1}, Q={inst['capacity']})",
                 # savepath=f"instance_{filename}.png")

    # 2) Résout et trace la meilleure solution
    solver = CVRPSolverVNSILS(inst, seed=seed)
    best_routes, best_cost = None, math.inf
    t0 = time.time()
    for _ in tqdm(range(runs), desc=f"VNS-ILS on {filename}"):
        sol, c = solver.solve(time_limit=time_limit,
                              restarts=vns_restarts,
                              ils_iters=vns_ils_iters,
                              shake_min=vns_shake_min,
                              shake_max=vns_shake_max)
        if solver.is_solution_valid(sol) and c < best_cost:
            best_routes, best_cost = sol, c
    t1 = time.time()

    if bks == 0:
        bks = best_cost

    print_solution(inst, best_routes, best_cost, bks, t1 - t0, filename)
    plot_solution(inst, best_routes,
                  title=f"{filename} (n={len(inst['demand'])-1}, Q={inst['capacity']})",
                  savepath=f"solution_{filename}.png")
    plt.show()

if __name__ == "__main__":
    # A-n32-k5 : réglages "normaux"
    run_and_plot(
        r"C:\Users\ameli\Documents\A3\Algo_combi_opti\Project_Algo\instances\A\A-n32-k5.vrp",
        runs=20,
        time_limit=5.0,
        seed=1000,
        vns_restarts=4,
        vns_ils_iters=500,
        vns_shake_min=1,
        vns_shake_max=4
    )

    # X-n101-k25 : version rapide
    run_and_plot(
        r"C:\Users\ameli\Documents\A3\Algo_combi_opti\Project_Algo\instances\M\M-n101-k10.vrp",
        runs=20,
        time_limit=3.0,
        seed=2000,
        vns_restarts=2,
        vns_ils_iters=400,
        vns_shake_min=1,
        vns_shake_max=3
    )
