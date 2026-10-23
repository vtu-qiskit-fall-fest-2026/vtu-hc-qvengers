
import csv
import random
from pathlib import Path

from classical_solver import solve_classical
from classical_heuristic import solve_heuristic
from quantum_solver import solve_max_cut


def generate_graph(num_vertices, probability, seed):
    rng = random.Random(seed)

    edges = [
        (u, v)
        for u in range(num_vertices)
        for v in range(u + 1, num_vertices)
        if rng.random() < probability
    ]

    if not edges:
        edges.append((0, 1))

    return edges


def graph_features(num_vertices, edges):
    degrees = [0] * num_vertices

    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    possible_edges = num_vertices * (num_vertices - 1) / 2

    return {
        "num_vertices": num_vertices,
        "num_edges": len(edges),
        "edge_density": len(edges) / possible_edges,
        "average_degree": sum(degrees) / num_vertices,
        "max_degree": max(degrees),
    }


def main():
    rows = []
    graph_id = 0

    for n in range(4, 9):
        for probability in (0.25, 0.5, 0.75):
            for repetition in range(2):
                graph_id += 1

                edges = generate_graph(
                    n,
                    probability,
                    seed=1000 + graph_id,
                )

                features = graph_features(n, edges)

                # Exact classical solver supplies the reference optimum.
                exact = solve_classical(n, edges)
                reference = exact["cut_value"]

                heuristic = solve_heuristic(
                    n,
                    edges,
                    seed=2000 + graph_id,
                )

                # Use the actual function from quantum_solver.py.
                qaoa = solve_max_cut(
                    n,
                    edges,
                    layers=1,
                    seed=3000 + graph_id,
                )

                for result in (exact, heuristic, qaoa):
                    gap = (
                        (reference - result["cut_value"]) / reference
                        if reference
                        else 0.0
                    )

                    rows.append({
                        "graph_id": graph_id,
                        **features,
                        "solver": result["algorithm"],
                        "cut_value": result["cut_value"],
                        "runtime_seconds": result["runtime_seconds"],
                        "optimality_gap": gap,
                        "reference_optimum": reference,
                    })

                print(
                    f"Graph {graph_id}: n={n}, "
                    f"exact={exact['cut_value']}, "
                    f"heuristic={heuristic['cut_value']}, "
                    f"QAOA={qaoa['cut_value']}"
                )

    output = Path("benchmark_all_results.csv")

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nBenchmark complete!")
    print(f"Graphs tested: {graph_id}")
    print(f"Solver results: {len(rows)}")
    print(f"Saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
