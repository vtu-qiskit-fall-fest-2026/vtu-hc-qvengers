
import csv
import random
from pathlib import Path
from time import perf_counter

import numpy as np
from classical_solver import solve_classical
from quantum_solver import solve_max_cut


def generate_graph(n, probability, rng):
    """Generate a random undirected graph."""
    edges = [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if rng.random() < probability
    ]

    # Ensure the graph has at least one edge.
    if not edges:
        edges = [(0, 1)]

    return edges


def graph_features(n, edges):
    """Calculate simple features available before choosing a solver."""
    degrees = [0] * n

    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    possible_edges = n * (n - 1) / 2

    return {
        "num_vertices": n,
        "num_edges": len(edges),
        "edge_density": len(edges) / possible_edges,
        "average_degree": sum(degrees) / n,
        "max_degree": max(degrees),
    }


def main():
    rng = random.Random(42)
    rows = []
    graph_id = 0

    # Keep exact search small because it checks exponentially many partitions.
    for n in range(4, 9):
        for density in (0.25, 0.50, 0.75):
            for repetition in range(4):
                graph_id += 1
                edges = generate_graph(n, density, rng)
                features = graph_features(n, edges)

                print(
                    f"\nGraph {graph_id}: n={n}, "
                    f"edges={len(edges)}, density={density}"
                )

                # Exact classical answer acts as the reference optimum.
                classical = solve_classical(n, edges)
                optimum = classical["cut_value"]

                # Run the QAOA simulator with a fixed seed.
                qaoa_start = perf_counter()
                quantum = solve_max_cut(
                    n, edges, layers=1, seed=42 + graph_id
                )
                qaoa_time = perf_counter() - qaoa_start

                print(
                    f"  Classical cut={optimum}, "
                    f"time={classical['runtime_seconds']:.6f}s"
                )
                print(
                    f"  QAOA cut={quantum['cut_value']}, "
                    f"time={qaoa_time:.6f}s"
                )

                for solver, result, runtime in (
                    (
                        "classical_exact",
                        classical,
                        classical["runtime_seconds"],
                    ),
                    ("qaoa_simulator", quantum, qaoa_time),
                ):
                    rows.append({
                        "graph_id": graph_id,
                        **features,
                        "solver": solver,
                        "cut_value": result["cut_value"],
                        "runtime_seconds": runtime,
                        "optimality_gap": optimum - result["cut_value"],
                        "reference_optimum": optimum,
                    })

    output_path = Path(__file__).with_name("benchmark_results.csv")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- BENCHMARK COMPLETE ---")
    print(f"Graph instances: {graph_id}")
    print(f"CSV records: {len(rows)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
