import os
from typing import List, Tuple
import random

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox

from vrp_io import load_instance
from aco_vrp import run_aco, run_aco_iter


def save_routes_plot(path: str, coords: List[Tuple[float, float]], routes: List[List[int]]):
    plt.figure(figsize=(8, 6))
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    plt.scatter(x[1:], y[1:], s=10, c="blue", label="Customers")
    plt.scatter([x[0]], [y[0]], s=60, c="red", label="Depot")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for idx, route in enumerate(routes):
        cx = [coords[i][0] for i in route]
        cy = [coords[i][1] for i in route]
        plt.plot(cx, cy, '-', linewidth=1.8, color=colors[idx % len(colors)], label=f"Route {idx+1}")
    plt.title("Best solution routes")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_pheromones_plot(path: str, coords: List[Tuple[float, float]], tau, top_percent: float):
    n = len(coords)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append(((i, j), tau[i][j]))
    if not edges:
        return
    edges.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(len(edges) * top_percent / 100.0))
    edges = edges[:k]
    max_tau = edges[0][1]
    min_tau = edges[-1][1]
    plt.figure(figsize=(8, 6))
    for (i, j), t in edges:
        x = [coords[i][0], coords[j][0]]
        y = [coords[i][1], coords[j][1]]
        if max_tau > min_tau:
            norm = (t - min_tau) / (max_tau - min_tau)
        else:
            norm = 1.0
        width = 0.5 + 4.5 * norm
        alpha = 0.2 + 0.8 * norm
        plt.plot(x, y, '-', linewidth=width, color=(1, 0, 0, alpha))
    plt.scatter([coords[0][0]], [coords[0][1]], s=60, c="red", label="Depot")
    plt.scatter([c[0] for c in coords[1:]], [c[1] for c in coords[1:]], s=10, c="black", label="Nodes")
    plt.title("Pheromone intensity (top edges)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_convergence_plot(path: str, history: List[float]):
    plt.figure(figsize=(7, 4))
    plt.plot(history, '-b')
    plt.title("Best-so-far over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def launch_gui():
    root = tk.Tk()
    root.title("VRP ACO (vrplib)")

    # Variables
    instance_var = tk.StringVar()
    # no output folder in UI
    alpha_var = tk.DoubleVar(value=1.0)
    beta_var = tk.DoubleVar(value=3.0)
    rho_var = tk.DoubleVar(value=0.1)
    ants_var = tk.IntVar(value=30)
    iters_var = tk.IntVar(value=200)
    # seed is internal random; not shown in UI
    best_known_var = tk.StringVar(value="")
    pher_top_var = tk.DoubleVar(value=5.0)
    two_opt_var = tk.BooleanVar(value=True)
    three_opt_var = tk.BooleanVar(value=False)

    def try_autofill_best_known(vrp_path: str):
        try:
            base, _ = os.path.splitext(vrp_path)
            sol_path = base + ".sol"
            if os.path.exists(sol_path):
                with open(sol_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.lower().startswith("cost"):
                            parts = line.split()
                            if len(parts) >= 2:
                                val = float(parts[1])
                                best_known_var.set(str(val))
                                return
        except Exception:
            pass

    controls: list[tk.Widget] = []

    # Matplotlib canvas for live view
    fig = plt.Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()

    def draw_points(coords):
        ax.clear()
        if not coords:
            canvas.draw()
            return
        x = [c[0] for c in coords]
        y = [c[1] for c in coords]
        ax.scatter(x[1:], y[1:], s=10, c="blue", label="Customers")
        ax.scatter([x[0]], [y[0]], s=60, c="red", label="Depot")
        ax.set_title("VRP Nodes")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        canvas.draw()
        root.update_idletasks()

    def draw_routes_and_pheromones(coords, routes, tau, top_percent):
        ax.clear()
        if not coords:
            canvas.draw()
            return
        x = [c[0] for c in coords]
        y = [c[1] for c in coords]
        ax.scatter(x[1:], y[1:], s=12, c="#1976D2", label="Customers")
        ax.scatter([x[0]], [y[0]], s=70, c="#D32F2F", label="Depot")
        # pheromone edges (top %)
        n = len(coords)
        edges = []
        if tau is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append(((i, j), tau[i][j]))
        if edges:
            edges.sort(key=lambda x: x[1], reverse=True)
            k = max(1, int(len(edges) * top_percent / 100.0))
            edges = edges[:k]
            max_tau = edges[0][1]
            min_tau = edges[-1][1]
            for (i, j), t in edges:
                xx = [coords[i][0], coords[j][0]]
                yy = [coords[i][1], coords[j][1]]
                if max_tau > min_tau:
                    norm = (t - min_tau) / (max_tau - min_tau)
                else:
                    norm = 1.0
                width = 0.5 + 4.5 * norm
                alpha = 0.12 + 0.68 * norm
                ax.plot(xx, yy, '-', linewidth=width, color=(0.8, 0.0, 0.0, alpha))
        # best routes (highlighted)
        for route in (routes or []):
            cx = [coords[i][0] for i in route]
            cy = [coords[i][1] for i in route]
            ax.plot(cx, cy, '-', linewidth=2.8, color="#00C853")
        ax.set_title("ACO (live): best solution in green, pheromones in red")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.tight_layout()
        canvas.draw()

    def set_controls_state(enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in controls:
            try:
                w.configure(state=state)
            except Exception:
                pass

    def browse_instance():
        path = filedialog.askopenfilename(title="Select VRP instance", filetypes=[("VRP/TSPLIB or Solomon", "*.vrp *.txt"), ("VRP", "*.vrp"), ("Solomon", "*.txt"), ("All files", "*.*")])
        if path:
            instance_var.set(path)
            try_autofill_best_known(path)
            set_controls_state(True)
            try:
                inst_preview = load_instance(path)
                if not inst_preview.coordinates or len(inst_preview.coordinates) < 2:
                    raise ValueError("Parsed 0 nodes from file. Check the file format.")
                draw_points(inst_preview.coordinates)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # removed output folder browse

    def run():
        try:
            instance_path = instance_var.get().strip()
            if not instance_path:
                messagebox.showerror("Error", "Please select a VRP instance file.")
                return
            # nothing to create; we render live only
            inst = load_instance(instance_path)
            if not inst.coordinates or len(inst.coordinates) < 2:
                raise ValueError("Parsed 0 nodes from file. Check the file format.")

            # Disable run during execution
            run_btn.configure(state="disabled")
            status_var.set("Running...")

            final_state = {"best_routes": None, "best_len": None}

            def iterate(gen):
                try:
                    it, best_routes, best_len, tau, hist = next(gen)
                    final_state["best_routes"] = best_routes
                    final_state["best_len"] = best_len
                    status_var.set(f"Iteration: {it}    Best: {best_len:.4f}")
                    draw_routes_and_pheromones(inst.coordinates, best_routes, tau, float(pher_top_var.get()))
                    root.after(1, iterate, gen)
                except StopIteration:
                    run_btn.configure(state="normal")
                    status_var.set("Finished")
                    # Show summary
                    best_routes = final_state["best_routes"] or []
                    best_len = final_state["best_len"] or 0.0
                    msg = [
                        f"Instance: {inst.name}",
                        f"Capacity: {inst.capacity} | Customers: {inst.dimension - 1} | Routes: {len(best_routes)}",
                        f"Best length: {best_len:.4f}",
                    ]
                    bk_str = best_known_var.get().strip()
                    if bk_str:
                        try:
                            bk = float(bk_str)
                            gap = 100.0 * (best_len - bk) / bk
                            msg.append(f"Gap vs best-known: {gap:.2f}% (best-known={bk})")
                        except Exception:
                            msg.append("Best-known value invalid (ignored)")
                    messagebox.showinfo("ACO finished", "\n".join(msg))

            gen = run_aco_iter(
                inst.distance_matrix,
                inst.demands,
                inst.capacity,
                iterations=int(iters_var.get()),
                num_ants=int(ants_var.get()),
                alpha=float(alpha_var.get()),
                beta=float(beta_var.get()),
                evaporation=float(rho_var.get()),
                seed=random.randint(0, 10**9),
                use_two_opt=bool(two_opt_var.get() or three_opt_var.get()),
                use_three_opt=bool(three_opt_var.get()),
            )
            # Start iteration loop
            root.after(1, iterate, gen)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Layout
    row = 0
    tk.Label(root, text="Instance (.vrp)").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    tk.Entry(root, textvariable=instance_var, width=60).grid(row=row, column=1, padx=6, pady=4)
    tk.Button(root, text="Browse", command=browse_instance).grid(row=row, column=2, padx=6, pady=4)
    row += 1

    # Removed output folder controls

    tk.Label(root, text="Alpha").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    alpha_entry = tk.Entry(root, textvariable=alpha_var, width=10)
    alpha_entry.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    controls.append(alpha_entry)
    row += 1

    tk.Label(root, text="Beta").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    beta_entry = tk.Entry(root, textvariable=beta_var, width=10)
    beta_entry.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    controls.append(beta_entry)
    row += 1

    tk.Label(root, text="Evaporation (rho)").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    rho_entry = tk.Entry(root, textvariable=rho_var, width=10)
    rho_entry.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    controls.append(rho_entry)
    row += 1

    tk.Label(root, text="Ants").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    ants_entry = tk.Entry(root, textvariable=ants_var, width=10)
    ants_entry.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    controls.append(ants_entry)
    row += 1

    tk.Label(root, text="Iterations").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    iters_entry = tk.Entry(root, textvariable=iters_var, width=10)
    iters_entry.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    controls.append(iters_entry)
    row += 1

    # Seed hidden (randomized internally)

    tk.Label(root, text="Best-known (optional)").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    best_known_entry = tk.Entry(root, textvariable=best_known_var, width=15)
    best_known_entry.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    controls.append(best_known_entry)
    row += 1

    tk.Label(root, text="Pheromone top %").grid(row=row, column=0, sticky="w", padx=6, pady=4)
    pher_entry = tk.Entry(root, textvariable=pher_top_var, width=10)
    pher_entry.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    controls.append(pher_entry)
    row += 1

    two_opt_cb = tk.Checkbutton(root, text="2-opt", variable=two_opt_var)
    two_opt_cb.grid(row=row, column=0, sticky="w", padx=6, pady=4)
    three_opt_cb = tk.Checkbutton(root, text="3-opt (slower)", variable=three_opt_var)
    three_opt_cb.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    controls.extend([two_opt_cb, three_opt_cb])
    row += 1

    status_var = tk.StringVar(value="Pick a VRP file to start")
    status_lbl = tk.Label(root, textvariable=status_var)
    status_lbl.grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=4)
    row += 1

    run_btn = tk.Button(root, text="Run ACO (Live)", command=run, bg="#4CAF50", fg="white")
    run_btn.grid(row=row, column=0, columnspan=3, pady=10)
    controls.append(run_btn)

    # Disable all parameter controls until a file is chosen
    set_controls_state(False)

    # Place live canvas
    row += 1
    canvas_widget.grid(row=row, column=0, columnspan=3, padx=6, pady=8, sticky="nsew")
    root.grid_rowconfigure(row, weight=1)
    root.grid_columnconfigure(1, weight=1)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()


