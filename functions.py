"""
helpers/functions.py

Small utility helpers used across the project:
- `get_vrptw_instance`: load a VRPLIB instance and its associated solution file.
- `show`: plot a VRP solution on a 2D map for visualization.

These helpers are intentionally lightweight: they perform I/O and plotting
and are designed to be called from scripts or notebooks that run inside the
project workspace. The implementations below focus on clarity rather than
performance and include comments explaining input/output expectations.
"""

import vrplib
import os
import matplotlib.pyplot as plt


def get_vrptw_instance(file_name):
    """
    This function return a dataset of the intance
    in the good format and his solution.

    EXECUTE ONLY IN THE scripts FOLDER ! with from functions import get_vrptw_instance
    
    Args:
        file_name (str): the name of the instance file that we want to use
    
    Return:
        tuple: (dataset, solution) where:
            - dataset (dict): Instance data (node_coord, capacity, etc.)
            - solution (dict): Solution with 'routes' and 'cost'
    """


    # Read the VRPLIB instance file using the `vrplib` helper. The returned
    # `data` dictionary contains keys such as 'node_coord', 'demand', 'capacity'
    # and sometimes an 'edge_weight' matrix (depending on the instance format).
    data = vrplib.read_instance(file_name, instance_format="vrplib")

    # Convention: the solution file is expected to live in the `instances`
    # folder alongside the .vrp files and uses the same basename with a
    # `.sol` extension. We build the path accordingly and ask `vrplib` to
    # parse it. If no solution file exists, `vrplib.read_solution` may raise
    # or return an empty structure depending on the library; callers should
    # handle that case if necessary.
    solution_path = os.path.join(file_name.replace(".vrp", ".sol"))
    solution = vrplib.read_solution(solution_path)

    # Return a tuple (data_set, data_set_solution) matching the rest of the
    # project code which expects both instance data and a reference solution.
    return data, solution

def show(solution, dataSet):
    """ 
    Plot the VRP solution on a 2D map.
    Args:
        solution (list[list[int]]): A VRP solution composed of several routes.
        Each route is a list of client indices assigned to a vehicle.
        
        dataSet (dict): Dictionary containing instance data.
        Must include:
            - 'node_coord': list of (x, y) coordinates for depot and clients.

    Description:
        The function displays the depot and clients on a scatter plot,
        labels each node with its index, and draws each vehicle route with
        a distinct color.

    Returns:
        None
    """
    # Extract coordinates for plotting
    coords = dataSet['node_coord']
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]

    plt.figure(figsize=(10, 8))

    # Plot depot (index 0) with a distinct marker and clients in another color
    plt.scatter(x[0], y[0], c='red', s=100, label='Depot', marker='s')
    plt.scatter(x[1:], y[1:], c='blue', s=70, label='Clients')

    # Annotate each node with its index to make routes easier to read
    for i, (xi, yi) in enumerate(coords):
        plt.text(xi, yi + 1, str(i), fontsize=9)

    # Colors for routes: use a qualitative colormap and wrap if there are
    # more vehicles than colors available.
    colors = plt.cm.tab20.colors

    # Draw each route as a polyline connecting nodes; we add depot at start
    # and end of the stored route for visualization purposes.
    for i, route in enumerate(solution):
        # Build the full route including the depot (index 0)
        full_route = [0] + route + [0]

        # Extract x/y coordinates for the points along this route
        route_coords = [coords[node] for node in full_route]
        rx = [c[0] for c in route_coords]
        ry = [c[1] for c in route_coords]

        color = colors[i % len(colors)]
        # Plot the polyline with small markers to show the visitation order
        plt.plot(rx, ry, '-o', color=color, label=f'Vehicle {i+1}', markersize=4, linewidth=1.5)

    # Labels and decorations
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    