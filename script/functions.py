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

    instance_path = os.path.join("..", "instances", file_name)
    data = vrplib.read_instance(instance_path, instance_format="solomon")

    solution_path = os.path.join("..", "instances", file_name.replace(".txt", ".sol"))
    solution = vrplib.read_solution(solution_path)

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
    coords = dataSet['node_coord']
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]

    plt.figure(figsize=(10, 8))

    plt.scatter(x[0], y[0], c='red', s=100, label='Dépôt', marker='s')
    plt.scatter(x[1:], y[1:], c='blue', s=70, label='Clients')

    for i, (xi, yi) in enumerate(coords):
        plt.text(xi, yi+1, str(i), fontsize=9)


    # Drawing the trucks solutions
    colors = plt.cm.tab20.colors

    for i, route in enumerate(solution):
        full_route = [0] + route + [0]

        route_coords = [coords[node] for node in full_route]
        rx = [c[0] for c in route_coords]
        ry = [c[1] for c in route_coords]

        color = colors[i % len(colors)]
        plt.plot(rx, ry, '-o', color=color, label=f'Vehicle {i+1}', markersize=4, linewidth=1.5)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    