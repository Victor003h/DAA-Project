"""
generator.py

Random instance generator for the
Degree-Constrained Minimum Spanning Tree (DC-MST) problem.
"""

import random
from typing import List, Tuple, Dict, Set


Edge = Tuple[int, int, float]


def generate_graph(num_vertices: int,
                   edge_probability: float,
                   weight_range: Tuple[int, int],
                   degree_bound: int,
                   **seed: int ):
    """
    Generates a random undirected weighted graph
    with uniform degree constraints.

    Parameters
    ----------
    num_vertices : int
        Number of vertices.
    edge_probability : float
        Probability of edge existence (Erdos-Renyi model).
    weight_range : tuple (min, max)
        Range for edge weights.
    degree_bound : int
        Maximum degree allowed for each vertex.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    vertices : set of int
    edges : list of Edge
    degree_bounds : dict
    """
    if seed is not None:
        random.seed(**seed)

    vertices: Set[int] = set(range(num_vertices))
    edges: List[Edge] = []

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < edge_probability:
                weight = random.randint(*weight_range)
                edges.append((i, j, float(weight)))

    degree_bounds: Dict[int, int] = {v: degree_bound for v in vertices}

    return vertices, edges, degree_bounds


def generate_feasible_instance(num_vertices: int,
                               edge_probability: float,
                               weight_range: Tuple[int, int],
                               degree_bound: int,
                               max_attempts: int = 100,
                               **seed: int):
    """
    Generates an instance that is likely to admit
    at least one feasible spanning tree.

    Retries generation if graph is too sparse.

    Returns
    -------
    vertices, edges, degree_bounds
    """
    for _ in range(max_attempts):
        vertices, edges, degree_bounds = generate_graph(
            num_vertices,
            edge_probability,
            weight_range,
            degree_bound,
            **seed
        )

        # A necessary (but not sufficient) condition:
        if len(edges) >= num_vertices - 1:
            return vertices, edges, degree_bounds

    raise RuntimeError("Failed to generate a feasible instance.")
