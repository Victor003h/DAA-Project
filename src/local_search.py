"""
local_search.py

Local search improvement algorithm for the
Degree-Constrained Minimum Spanning Tree (DC-MST) problem.

This algorithm improves an initial feasible solution
using edge-swap neighborhood exploration.
"""

from typing import List, Tuple, Dict, Set
from collections import defaultdict, deque
from src.utils.graph import (
    tree_cost,
    build_adjacency,
    respects_degree_constraints
    )

Edge = Tuple[int, int, float]   # (u, v, weight)
Tree = List[Edge]

def connected_components_after_removal(vertices: Set[int],
                                       tree: Tree,
                                       removed_edge: Edge) -> List[Set[int]]:
    """
    Returns the connected components obtained after removing one edge.
    """
    u, v, _ = removed_edge
    remaining_edges = [e for e in tree if e != removed_edge]

    adj = build_adjacency(vertices, remaining_edges)

    visited = set()
    components = []

    for start in vertices:
        if start not in visited:
            comp = set()
            queue = deque([start])
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    comp.add(current)
                    queue.extend(adj[current] - visited)
            components.append(comp)

    return components


def local_search_dc_mst(vertices: Set[int],
                        edges: List[Edge],
                        initial_tree: Tree,
                        degree_bounds: Dict[int, int]) -> Tree:
    """
    Local search algorithm for DC-MST.

    Parameters
    ----------
    vertices : set of int
        Vertex identifiers.
    edges : list of Edge
        All possible edges.
    initial_tree : Tree
        Initial feasible solution.
    degree_bounds : dict
        Maximum allowed degree for each vertex.

    Returns
    -------
    Tree
        Improved solution (local optimum).
    """
    current_tree = initial_tree[:]
    current_cost = tree_cost(current_tree)

    improved = True
    while improved:
        improved = False

        for edge_in in list(current_tree):
            components = connected_components_after_removal(
                vertices, current_tree, edge_in
            )

            if len(components) != 2:
                continue

            comp_a, comp_b = components

            for u, v, weight in edges:
                if (u in comp_a and v in comp_b) or (u in comp_b and v in comp_a):
                    if (u, v, weight) in current_tree or (v, u, weight) in current_tree:
                        continue

                    candidate_tree = current_tree[:]
                    candidate_tree.remove(edge_in)
                    candidate_tree.append((u, v, weight))

                    if not respects_degree_constraints(candidate_tree, degree_bounds):
                        continue

                    candidate_cost = tree_cost(candidate_tree)

                    if candidate_cost < current_cost:
                        current_tree = candidate_tree
                        current_cost = candidate_cost
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

    return current_tree
