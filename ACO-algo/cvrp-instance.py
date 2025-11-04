import vrplib
import matplotlib
import math


class CVRPInstance:
    def __init__(self, file_path):
        self.instance = vrplib.load(file_path)
        self.num_customers = len(self.instance.customers)
        self.depot = self.instance.depot
        self.vehicle_capacity = self.instance.vehicle_capacity
        self.customer_demands = [customer.demand for customer in self.instance.customers]
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        locations = [self.depot] + [customer.location for customer in self.instance.customers]
        num_locations = len(locations)
        distance_matrix = [[0] * num_locations for _ in range(num_locations)]

        for i in range(num_locations):
            for j in range(num_locations):
                if i != j:
                    distance_matrix[i][j] = math.dist(locations[i], locations[j])
        return distance_matrix
    
    