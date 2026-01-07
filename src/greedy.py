"""
greedy.py

Greedy heuristic for the Degree-Constrained
Minimum Spanning Tree (DC-MST) problem.

This implementation is inspired by Kruskal's algorithm,
extended with explicit degree constraints.
"""

from typing import List, Tuple, Dict, Set
from src.utils.union_find import UnionFind

Edge = Tuple[int, int, float]   # (u, v, weight)
Tree = List[Edge]



def greedy_dc_mst(vertices: Set[int],
                  edges: List[Edge],
                  degree_bounds: Dict[int, int]) -> Tree:
    """
    Greedy solver for DC-MST.

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
        Degree-constrained spanning tree (heuristic).
    """
    sorted_edges = sorted(edges, key=lambda e: e[2])
    uf = UnionFind(vertices)

    degree = {v: 0 for v in vertices}
    tree: Tree = []

    for u, v, weight in sorted_edges:
        if uf.find(u) != uf.find(v):
            if degree[u] < degree_bounds[u] and degree[v] < degree_bounds[v]:
                tree.append((u, v, weight))
                uf.union(u, v)
                degree[u] += 1
                degree[v] += 1

        if len(tree) == len(vertices) - 1:
            break

    return tree
