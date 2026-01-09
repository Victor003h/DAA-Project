import math
import random
from copy import deepcopy

from utils.graph import (

    respects_degree_constraints_v2,
    total_cost,
    generate_neighbor
)


def simulated_annealing(
    graph,
    degree_bounds,
    initial_solution,
    initial_temperature=1000.0,
    cooling_rate=0.995,
    min_temperature=1e-3,
    max_iterations=10000
):
    """
    Simulated Annealing for Degree-Constrained MST.

    Parameters
    ----------
    graph : Graph
        Original graph.
    degree_bounds : dict
        Maximum degree for each vertex.
    initial_solution : set
        Initial feasible spanning tree.
    """

    current_solution = deepcopy(initial_solution)
    best_solution = deepcopy(initial_solution)

    current_cost = total_cost(graph, current_solution)
    best_cost = current_cost

    temperature = initial_temperature
    iteration = 0

    while temperature > min_temperature and iteration < max_iterations:
        neighbor = generate_neighbor(graph, current_solution)

        if neighbor is None:
            iteration += 1
            temperature *= cooling_rate
            continue

        if not respects_degree_constraints_v2(neighbor, degree_bounds):
            iteration += 1
            temperature *= cooling_rate
            continue

        neighbor_cost = total_cost(graph, neighbor)
        delta = neighbor_cost - current_cost

        if delta < 0:
            current_solution = neighbor
            current_cost = neighbor_cost
        else:
            acceptance_prob = math.exp(-delta / temperature)
            if random.random() < acceptance_prob:
                current_solution = neighbor
                current_cost = neighbor_cost

        if current_cost < best_cost:
            best_solution = deepcopy(current_solution)
            best_cost = current_cost

        temperature *= cooling_rate
        iteration += 1

    return best_solution, best_cost
