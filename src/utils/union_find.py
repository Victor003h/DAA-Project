"""
graph.py

Utility functions for graph and tree operations
used in the DC-MST project.
"""

from typing import List, Tuple, Dict, Set
from collections import defaultdict, deque


Edge = Tuple[int, int, float]
Tree = List[Edge]


def tree_cost(tree: Tree) -> float:
    """
    Computes the total weight of a tree.
    """
    return sum(weight for _, _, weight in tree)


def compute_degrees(tree: Tree) -> Dict[int, int]:
    """
    Computes the degree of each vertex in a tree.
    """
    degrees = defaultdict(int)
    for u, v, _ in tree:
        degrees[u] += 1
        degrees[v] += 1
    return degrees


def respects_degree_constraints(tree: Tree,
                                degree_bounds: Dict[int, int]) -> bool:
    """
    Checks whether a tree respects all degree constraints.
    """
    degrees = compute_degrees(tree)
    for v, deg in degrees.items():
        if deg > degree_bounds[v]:
            return False
    return True


def build_adjacency(vertices: Set[int],
                    edges: Tree) -> Dict[int, Set[int]]:
    """
    Builds adjacency list from edges.
    """
    adj = {v: set() for v in vertices}
    for u, v, _ in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def is_connected(vertices: Set[int], edges: Tree) -> bool:
    """
    Checks whether the graph defined by edges is connected.
    """
    if not vertices:
        return True

    adj = build_adjacency(vertices, edges)
    visited = set()
    queue = deque([next(iter(vertices))])

    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            queue.extend(adj[current] - visited)

    return visited == vertices
