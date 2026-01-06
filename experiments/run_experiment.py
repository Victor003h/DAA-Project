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

    # --- Brute Force (only for small instances) ---
    if config["num_vertices"] <= config["max_exact_n"]:
        start = time.perf_counter()
        exact_tree = brute_force_dc_mst(vertices, edges, degree_bounds)
        exact_time = time.perf_counter() - start

        results["exact_cost"] = tree_cost(exact_tree)
        results["exact_time"] = exact_time

        results["approx_ratio_greedy"] = (
            results["greedy_cost"] / results["exact_cost"]
        )
        results["approx_ratio_local"] = (
            results["local_cost"] / results["exact_cost"]
        )
    else:
        results["exact_cost"] = None
        results["exact_time"] = None
        results["approx_ratio_greedy"] = None
        results["approx_ratio_local"] = None

    return results


def run_experiments(configs: List[Dict], output_file: str):
    """
    Runs all experiments and saves results to CSV.
    """
    if not configs:
        return

    fieldnames = list(run_single_experiment(configs[0]).keys())

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for config in configs:
            result = run_single_experiment(config)
            writer.writerow(result)


if __name__ == "__main__":
    experiment_configs = [
        {
            "num_vertices": 6,
            "edge_probability": 0.6,
            "weight_range": (1, 20),
            "degree_bound": 2,
            "seed": 42,
            "max_exact_n": 8
        },
        {
            "num_vertices": 8,
            "edge_probability": 0.6,
            "weight_range": (1, 20),
            "degree_bound": 3,
            "seed": 43,
            "max_exact_n": 8
        },
        {
            "num_vertices": 10,
            "edge_probability": 0.7,
            "weight_range": (1, 20),
            "degree_bound": 3,
            "seed": 44,
            "max_exact_n": 8
        }
    ]

    run_experiments(experiment_configs, "results.csv")
