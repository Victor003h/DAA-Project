"""
graph.py

Utility functions for graph and tree operations
used in the DC-MST project.
"""

from typing import List, Tuple, Dict, Set
from collections import defaultdict, deque
import random

Edge = Tuple[int, int]



def total_cost(graph, tree_edges):
    """
    Computes the total weight of a spanning tree.
    """
    cost = 0
    for u, v in tree_edges:
        cost += graph["weights"][(u, v)]
    return cost



def compute_degrees(tree: list[Edge]) -> Dict[int, int]:
    """
    Computes the degree of each vertex in a tree.
    """
    degrees = defaultdict(int)
    for u, v in tree:
        degrees[u] += 1
        degrees[v] += 1
    return degrees



def respects_degree_constraints(tree_edges, degree_bounds):
    """
    Checks if all vertices respect their degree constraints.
    """
    degree = defaultdict(int)

    for u, v in tree_edges:
        degree[u] += 1
        degree[v] += 1

        if degree[u] > degree_bounds[u]:
            return False
        if degree[v] > degree_bounds[v]:
            return False

    return True


def build_adjacency(vertices: Set[int],
                    edges: list[Edge]) -> Dict[int, Set[int]]:
    """
    Builds adjacency list from edges.
    """
    adj = {v: set() for v in vertices}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def is_connected(vertices: Set[int], edges: list[Edge]) -> bool:
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

def connected_components(vertices, edges):
    """
    Returns connected components of a graph.
    """
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    components = []

    for v in vertices:
        if v not in visited:
            stack = [v]
            comp = set()
            visited.add(v)

            while stack:
                x = stack.pop()
                comp.add(x)
                for y in adj[x]:
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)

            components.append(comp)

    return components

def generate_neighbor(graph, tree_edges):
    """
    Generates a neighbor solution by edge exchange.
    """
    vertices = graph["vertices"]
    all_edges = set(graph["edges"])

    tree_edges = set(tree_edges)

    removed_edge = random.choice(tuple(tree_edges))
    new_tree = tree_edges - {removed_edge}

    components = connected_components(vertices, new_tree)
    if len(components) != 2:
        return None

    comp_a, comp_b = components

    candidate_edges = [
        (u, v) for (u, v) in all_edges
        if (u in comp_a and v in comp_b) or
           (u in comp_b and v in comp_a)
    ]

    if not candidate_edges:
        return None

    added_edge = random.choice(candidate_edges)
    new_tree.add(added_edge)

    return new_tree


def generate_neighbor_with_move(graph, tree_edges):
    """
    Generates neighbors along with their corresponding move.
    """
    vertices = graph["vertices"]
    all_edges = set(graph["edges"])
    tree_edges = set(tree_edges)

    for removed_edge in tree_edges:
        partial_tree = tree_edges - {removed_edge}
        components = connected_components(vertices, partial_tree)

        if len(components) != 2:
            continue

        comp_a, comp_b = components

        for edge in all_edges:
            u, v = edge
            if (u in comp_a and v in comp_b) or \
               (u in comp_b and v in comp_a):

                new_tree = partial_tree | {edge}
                move = (removed_edge, edge)

                yield new_tree, move


def is_tree(vertices, tree_edges):
    """
    Checks if a set of edges forms a spanning tree.
    """
    if len(tree_edges) != len(vertices) - 1:
        return False

    parent = {v: v for v in vertices}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    for u, v in tree_edges:
        if not union(u, v):
            return False

    root = find(vertices[0])
    return all(find(v) == root for v in vertices)
