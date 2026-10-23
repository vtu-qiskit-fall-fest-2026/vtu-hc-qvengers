
from itertools import product
from time import perf_counter


def cut_value(partition, edges):
    """Count edges whose endpoints belong to different groups."""
    return sum(partition[u] != partition[v] for u, v in edges)


def solve_classical(num_vertices, edges):
    """Find the exact optimal Max-Cut by checking all partitions."""
    start = perf_counter()

    if num_vertices < 2:
        raise ValueError("Max-Cut requires at least two vertices.")

    edges = list(edges)

    if not edges:
        raise ValueError("The graph must contain at least one edge.")

    best_partition = None
    best_value = -1

    # Fix vertex 0 to group 0 to avoid checking duplicate complements.
    for remaining in product((0, 1), repeat=num_vertices - 1):
        partition = (0,) + remaining
        value = cut_value(partition, edges)

        if value > best_value:
            best_partition = partition
            best_value = value

    return {
        "algorithm": "Exact classical search",
        "partition": best_partition,
        "cut_value": best_value,
        "runtime_seconds": round(perf_counter() - start, 6),
    }


if __name__ == "__main__":
    graph_edges = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
    ]

    print("Testing classical solver on a 4-vertex graph...")
    for key, value in solve_classical(4, graph_edges).items():
        print(f"{key}: {value}")
