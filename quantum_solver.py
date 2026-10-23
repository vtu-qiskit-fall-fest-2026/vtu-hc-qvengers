
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from time import perf_counter

def cut_value(partition, edges):
    return sum(partition[u] != partition[v] for u, v in edges)


def solve_max_cut(num_vertices, edges, layers=1, seed=42):
    start = perf_counter()
    rng = np.random.default_rng(seed)

    def circuit_state(params):
        gammas = params[:layers]
        betas = params[layers:]

        circuit = QuantumCircuit(num_vertices)
        circuit.h(range(num_vertices))

        for gamma, beta in zip(gammas, betas):
            for u, v in edges:
                circuit.rzz(-gamma, u, v)
            circuit.rx(2 * beta, range(num_vertices))

        return Statevector.from_instruction(circuit)

    def expected_cut(params):
        probabilities = circuit_state(params).probabilities()
        return sum(
            probability * cut_value(
                tuple((i >> q) & 1 for q in range(num_vertices)),
                edges,
            )
            for i, probability in enumerate(probabilities)
        )

    initial = rng.uniform(0, np.pi, 2 * layers)
    result = minimize(
        lambda params: -expected_cut(params),
        initial,
        method="COBYLA",
        options={"maxiter": 80},
    )

    probabilities = circuit_state(result.x).probabilities()
    
    # Sample bitstrings according to the QAOA state probabilities.
    shots = 2048
    sampled_indices = rng.choice(
        len(probabilities),
        size=shots,
        p=probabilities,
    )

    best_partition = None
    best_value = -1

    for i in sampled_indices:
        partition = tuple(
            (int(i) >> q) & 1
            for q in range(num_vertices)
        )
        value = cut_value(partition, edges)

        if value > best_value:
            best_partition = partition
            best_value = value


    for i, probability in enumerate(probabilities):
        partition = tuple(
            (i >> q) & 1 for q in range(num_vertices)
        )
        value = cut_value(partition, edges)

        if value > best_value:
            best_partition = partition
            best_value = value

    return {
        "algorithm": "QAOA (statevector simulator)",
        "partition": best_partition,
        "cut_value": best_value,
        "expected_cut": round(expected_cut(result.x), 4),
        "runtime_seconds": round(perf_counter() - start, 4),
    }


if __name__ == "__main__":
    graph_edges = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
    ]

    print("Testing QAOA on a 4-vertex graph...")
    for key, value in solve_max_cut(4, graph_edges).items():
        print(f"{key}: {value}")
