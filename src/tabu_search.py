from collections import deque
from copy import deepcopy

from src.utils.graph import (
    respects_degree_constraints,
    total_cost,
    generate_neighbor_with_move
)


def tabu_search(
    graph,
    degree_bounds,
    initial_solution,
    tabu_tenure=20,
    max_iterations=5000
):
    """
    Tabu Search for Degree-Constrained MST.
    """

    current_solution = deepcopy(initial_solution)
    best_solution = deepcopy(initial_solution)

    current_cost = total_cost(graph, current_solution)
    best_cost = current_cost

    tabu_list = deque(maxlen=tabu_tenure)

    for _ in range(max_iterations):
        best_candidate = None
        best_candidate_cost = float("inf")
        best_move = None

        for neighbor, move in generate_neighbor_with_move(graph, current_solution):
            if move in tabu_list:
                continue

            if not respects_degree_constraints(neighbor, degree_bounds):
                continue

            cost = total_cost(graph, neighbor)

            if cost < best_candidate_cost:
                best_candidate = neighbor
                best_candidate_cost = cost
                best_move = move

        if best_candidate is None:
            break

        current_solution = best_candidate
        current_cost = best_candidate_cost
        tabu_list.append(best_move)

        if current_cost < best_cost:
            best_solution = deepcopy(current_solution)
            best_cost = current_cost

    return best_solution, best_cost
