"""
run_experiments.py

Experimental evaluation of algorithms for the
Degree-Constrained Minimum Spanning Tree (DC-MST) problem.
"""

import time
import csv
from typing import List, Dict

from instances.generator import generate_feasible_instance
from src.brute_force import brute_force_dc_mst
from src.greedy import greedy_dc_mst
from src.local_search import local_search_dc_mst
from src.utils.graph import tree_cost


def run_single_experiment(config: Dict) -> Dict:
    """
    Runs one experiment instance and measures performance.
    """
    vertices, edges, degree_bounds = generate_feasible_instance(
        num_vertices=config["num_vertices"],
        edge_probability=config["edge_probability"],
        weight_range=config["weight_range"],
        degree_bound=config["degree_bound"],
        seed=config["seed"]
    )

    results = {
        "num_vertices": config["num_vertices"],
        "degree_bound": config["degree_bound"],
        "edge_probability": config["edge_probability"]
    }

    # --- Greedy ---
    start = time.perf_counter()
    greedy_tree = greedy_dc_mst(vertices, edges, degree_bounds)
    greedy_time = time.perf_counter() - start

    results["greedy_cost"] = tree_cost(greedy_tree)
    results["greedy_time"] = greedy_time

    # --- Local Search ---
    start = time.perf_counter()
    local_tree = local_search_dc_mst(
        vertices, edges, greedy_tree, degree_bounds
    )
    local_time = time.perf_counter() - start

    results["local_cost"] = tree_cost(local_tree)
    results["local_time"] = local_time

    # --- Brute
