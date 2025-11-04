import matplotlib.pyplot as plt
import math
import numpy as np
import random


class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def random_solution(self):
        solution = random.sample(self.all_inds, len(self.all_inds))
        return solution

    def get_solution(self, ant):
        solution = []
        visited = set()
        for _ in range(len(self.distances)):
            move = self._pick_move(ant, visited)
            solution.append(move)
            visited.add(move)
        return solution
    
    def _pick_move(self, ant, visited):
        pheromone = np.copy(self.pheromone[ant][-1])
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * (( 1.0 / self.distances[ant][-1]) ** self.beta)
        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move
    
    def _spread_pheromone(self, all_solutions, n_best):
        sorted_solutions = sorted(all_solutions, key=lambda x: x[1])
        for solution, dist in sorted_solutions[:n_best]:
            for move in solution:
                self.pheromone[move] += 1.0 / self.distances[move]

    
    