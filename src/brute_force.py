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


Edge = Tuple[int, int, float]   # (u, v, weight)
Tree = List[Edge]


def is_connected(vertices: Set[int], edges: Tree) -> bool:
    """
    Checks if the graph defined by 'edges' is connected.
    """
    if not vertices:
        return True

    adjacency = {v: set() for v in vertices}
    for u, v, _ in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    visited = set()
    stack = [next(iter(vertices))]

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            stack.extend(adjacency[current] - visited)

    return visited == vertices


def has_cycle(vertices: Set[int], edges: Tree) -> bool:
    """
    Checks whether the given edges contain a cycle
    using Union-Find logic.
    """
    parent = {v: v for v in vertices}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[ry] = rx
        return True

    for u, v, _ in edges:
        if not union(u, v):
            return True

    return False


def respects_degree_constraints(edges: Tree,
                                degree_bounds: Dict[int, int]) -> bool:
    """
    Checks if degree constraints are respected.
    """
    degrees = {v: 0 for v in degree_bounds}

    for u, v, _ in edges:
        degrees[u] += 1
        degrees[v] += 1
        if degrees[u] > degree_bounds[u] or degrees[v] > degree_bounds[v]:
            return False

    return True


def tree_cost(edges: Tree) -> float:
    """
    Computes the total cost of a tree.
    """
    return sum(weight for _, _, weight in edges)


def brute_force_dc_mst(vertices: Set[int],
                       edges: List[Edge],
                       degree_bounds: Dict[int, int]) -> Tree:
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

        cost = tree_cost(candidate)
        if cost < best_cost:
            best_cost = cost
            best_tree = candidate

    if best_tree is None:
        raise ValueError("No feasible degree-constrained spanning tree found.")
    return best_tree
