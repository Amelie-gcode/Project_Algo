import os
import time
from CW import * 

def display_results(filepath, best_cost, elapsed_time):
    """
    Display CVRP/VRPTW instance performance summary.
    Automatically extracts instance name from filepath,
    looks up known optimal cost, and computes gap (%).
    """

    # --- Extract instance name from path ---
    instance_name = os.path.splitext(os.path.basename(filepath))[0]

    # --- Known optima for benchmark instances ---

    
    opt_cost=total_travel_distance(BEST_SOLUTION)

    gap = None

    if opt_cost:
        gap = 100 * (best_cost - opt_cost) / opt_cost

    # --- Print formatted result table ---
    print("\nRESULT SUMMARY")
    print(f"{'Instance':<15}{'Optimum':<12}{'Your cost':<12}{'Gap (%)':<10}{'Time (s)':<10}")
    print("-" * 60)
    print(f"{instance_name:<15}"
          f"{(opt_cost if opt_cost else 'N/A'):<12.2f}"
          f"{best_cost:<12.2f}"
          f"{(gap if gap is not None else 'N/A'):<10.2f}"
          f"{elapsed_time:<10.2f}")

    # --- Optional return for logging / saving results ---
    return {
        "instance": instance_name,
        "optimum": opt_cost,
        "your_cost": best_cost,
        "gap_percent": gap,
        "time_sec": elapsed_time
    }
