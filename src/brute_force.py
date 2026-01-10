"""
brute_force.py

Exact (brute-force) algorithm for the Degree-Constrained
Minimum Spanning Tree (DC-MST) problem.

This implementation enumerates all possible spanning trees,
checks degree constraints, and selects the minimum-cost solution.

Intended for small instances only.
"""

import itertools
from typing import List, Tuple, Dict, Set
from src.utils.union_find import UnionFind
from src.utils.graph import (
    is_connected,
    respects_degree_constraints,
    total_cost,
    Edge,
    
    
)



def has_cycle(vertices: Set[int], edges: List[Edge]) -> bool:
    """
    Checks whether the given edges contain a cycle
    using Union-Find logic.
    """
    union_find = UnionFind(vertices)   
        
    for u, v in edges:
        if not union_find.union(u, v):
            return True

    return False


def brute_force_dc_mst(graph,
                       degree_bounds: Dict[int, int]) -> Tuple[list, float]:
    """
    Brute-force solver for DC-MST.

    Parameters
    ----------
    vertices : set of int
        Vertex identifiers.
    edges : list of Edge
        All possible edges.
    degree_bounds : dict
        Maximum allowed degree for each vertex.

    Returns
    -------
    Tree
        Optimal degree-constrained spanning tree.
    """
    vertices = graph["vertices"]
    edges = graph["edges"]
    n = len(vertices)
    best_tree = None
    best_cost = float("inf")

    for candidate in itertools.combinations(edges, n - 1):
        candidate = list(candidate)

        if has_cycle(vertices, candidate):
            continue

        if not is_connected(vertices, candidate):
            continue

        if not respects_degree_constraints(candidate, degree_bounds):
            continue

        cost = total_cost(graph, candidate)
        if cost < best_cost:
            best_cost = cost
            best_tree = candidate

    if best_tree is None:
        raise ValueError("No feasible degree-constrained spanning tree found.")
    return best_tree,best_cost
