"""
run_experiments.py

Experimental evaluation of algorithms for the
Degree-Constrained Minimum Spanning Tree (DC-MST) problem.
"""

import time
import csv
from typing import List, Dict,Tuple

from instances.generator import generate_feasible_instance
from src.brute_force import brute_force_dc_mst
from src.greedy import greedy_dc_mst
from src.local_search import local_search_dc_mst
from src.simulated_annealing import simulated_annealing
from src.tabu_search import tabu_search

ALGORITHMS = {
    "BruteForce": brute_force_dc_mst,
    "Greedy": greedy_dc_mst,
    "LocalSearch": local_search_dc_mst,
    "SimulatedAnnealing": simulated_annealing,
    "TabuSearch": tabu_search
}

def density(num_vertices: int, num_edges: int) -> float:
    """
    Calculates the density of a graph.
    """
    if num_vertices <= 1:
        return 0.0
    max_edges = num_vertices * (num_vertices - 1) / 2
    return num_edges / max_edges


def run_single_experiment(config: Dict) -> list[Dict]:
    """
    Runs one experiment instance and measures performance.
    """
    vertices, edges,weights, degree_bounds = generate_feasible_instance(
        num_vertices=config["num_vertices"],
        edge_probability=config["edge_probability"],
        weight_range=config["weight_range"],
        degree_bound=config["degree_bound"],
        seed=config["seed"]
    )
    density_value = density(len(vertices), len(edges)) 
       
    instance = {
        "vertices": vertices,
        "edges": edges,
        "weights": weights
    }

    results = []

    # Initial solution for heuristics and metaheuristics
    initial_solution,_= greedy_dc_mst(
        instance, degree_bounds
    )

    for name, algorithm in ALGORITHMS.items():
        start = time.time()

        if name  in ("BruteForce", "Greedy"):
            solution, cost = algorithm(
                instance, 
                degree_bounds
            )
    
        else:  # Metaheuristics and local search
            solution, cost = algorithm(
                instance,
                degree_bounds,
                initial_solution
            )

        elapsed = time.time() - start

        results.append( {
            "Algorithm": name,
            "Vertices": vertices,
            "Density": density_value,
            "DegreeBounds": degree_bounds,
            "Cost": cost,
            "Time": elapsed
        })

    return results
def run_experiments(configs: List[Dict], output_file: str):
    """
    Runs all experiments and saves results to a single CSV file.
    Each row corresponds to one algorithm executed on one instance.
    """

    if not configs:
        return

    # Run once to infer CSV schema
    first_results = run_single_experiment(configs[0])

    if not first_results:
        return

    fieldnames = list(first_results[0].keys())

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for config in configs:
            results = run_single_experiment(config)
            
            for result in results:
                writer.writerow(result)
                
        f.flush()

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
      
    ]

    run_experiments(experiment_configs, "results.csv")
    