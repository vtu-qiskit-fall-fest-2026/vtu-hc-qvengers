
from random import Random
from time import perf_counter


def cut_value(partition, edges):
    """Count edges whose endpoints belong to different groups."""
    return sum(
        partition[u] != partition[v]
        for u, v in edges
    )


def solve_heuristic(num_vertices, edges, seed=42, max_passes=100):
    """Find a good Max-Cut partition using greedy local search."""

    if num_vertices < 2:
        raise ValueError("Max-Cut requires at least two vertices.")

    edges = list(edges)

    if not edges:
        raise ValueError("The graph must contain at least one edge.")

    for u, v in edges:
        if not (0 <= u < num_vertices and 0 <= v < num_vertices):
            raise ValueError("Edge contains an invalid vertex index.")
        if u == v:
            raise ValueError("Self-loops are not supported.")

    start = perf_counter()
    rng = Random(seed)

    # Assign each vertex randomly to group 0 or group 1.
    partition = [
        rng.randint(0, 1)
        for _ in range(num_vertices)
    ]

    # Store the neighbors of every vertex.
    neighbors = [[] for _ in range(num_vertices)]

    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)

    # Move vertices whenever doing so improves the cut.
    for _ in range(max_passes):
        improved = False
        vertex_order = list(range(num_vertices))
        rng.shuffle(vertex_order)

        for vertex in vertex_order:
            current_group = partition[vertex]

            gain = sum(
                1 if partition[neighbor] == current_group else -1
                for neighbor in neighbors[vertex]
            )

            if gain > 0:
                partition[vertex] = 1 - current_group
                improved = True

        if not improved:
            break

    elapsed = perf_counter() - start

    return {
        "algorithm": "Greedy local search",
        "partition": tuple(partition),
        "cut_value": cut_value(partition, edges),
        "runtime_seconds": round(elapsed, 6),
    }


if __name__ == "__main__":
    graph_edges = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
    ]

    print("Testing classical heuristic on a 4-vertex graph...")

    result = solve_heuristic(4, graph_edges)

    for key, value in result.items():
        print(f"{key}: {value}")
