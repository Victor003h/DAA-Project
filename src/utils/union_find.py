"""
union_find.py

Disjoint Set Union (Union-Find) data structure.
Used for cycle detection and connectivity checks.
"""

from typing import Set


class UnionFind:
    """
    Union-Find data structure with path compression.
    """

    def __init__(self, vertices: Set[int]):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, x: int) -> int:
        """
        Finds the representative of x with path compression.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Unites the sets of x and y.
        Returns False if x and y were already connected.
        """
        rx, ry = self.find(x), self.find(y)

        if rx == ry:
            return False

        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

        return True
