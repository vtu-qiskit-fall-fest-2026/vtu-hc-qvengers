
from classical_solver import solve_classical
from classical_heuristic import solve_heuristic
from quantum_solver import solve_max_cut

from ml_selector import (
    train_runtime_models,
    recommend_solver as ml_recommend_solver,
)


def get_graph_features(num_vertices, edges):
    edges = list(edges)
    num_edges = len(edges)

from classical_solver import solve_classical
from classical_heuristic import solve_heuristic
from quantum_solver import solve_max_cut

from ml_selector import (
    train_runtime_models,
    recommend_solver as ml_recommend_solver,
)


# Conservative limits; validate with benchmarks before raising.
MAX_EXACT_VERTICES = 16
MAX_QAOA_VERTICES = 16
MAX_GREEDY_VERTICES = 100


def get_graph_features(num_vertices, edges):
    edges = list(edges)
    num_edges = len(edges)

    possible_edges = num_vertices * (num_vertices - 1) / 2
    edge_density = (
        num_edges / possible_edges if possible_edges else 0.0
    )

    degrees = [0] * num_vertices
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    return {
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "edge_density": edge_density,
        "average_degree": (
            sum(degrees) / num_vertices if num_vertices else 0.0
        ),
        "max_degree": max(degrees, default=0),
    }


def run_all_solvers(num_vertices, edges, seed=42):
    if num_vertices > MAX_EXACT_VERTICES:
        raise ValueError(
            f"Compare All supports up to {MAX_EXACT_VERTICES} vertices "
            "because exact search becomes expensive. For larger graphs, "
            "use Fastest to run the scalable greedy solver."
        )

    if num_vertices > MAX_QAOA_VERTICES:
        raise ValueError(
            f"QAOA simulation is limited to {MAX_QAOA_VERTICES} vertices."
        )

    edges = list(edges)
    return [
        solve_classical(num_vertices, edges),
        solve_heuristic(num_vertices, edges, seed=seed),
        solve_max_cut(num_vertices, edges, layers=1, seed=seed),
    ]


def recommend_solver(
    num_vertices,
    edges,
    objective="fastest",
    seed=42,
    models=None,
):
    valid_objectives = {"fastest", "quality", "quantum", "compare"}

    if objective not in valid_objectives:
        raise ValueError(
            f"Invalid objective: {objective}. "
            f"Choose from {sorted(valid_objectives)}."
        )

    edges = list(edges)
    predicted_times = None

    # For larger graphs, avoid ML extrapolation and expensive solvers.
    if objective == "fastest" and num_vertices > MAX_EXACT_VERTICES:
        selected = solve_heuristic(num_vertices, edges, seed=seed)
        results = [selected]
        reason = (
            "Large graph: selected greedy local search to avoid "
            "expensive exact search and QAOA statevector simulation. "
            "The ML model was trained only on small graphs."
        )

    elif objective == "fastest":
        if models is None:
            models, _ = train_runtime_models()

        features = get_graph_features(num_vertices, edges)
        ml_choice, predicted_times = ml_recommend_solver(
            models, **features
        )

        if ml_choice == "classical_exact":
            selected = solve_classical(num_vertices, edges)
        elif ml_choice == "qaoa_simulator":
            if num_vertices > MAX_QAOA_VERTICES:
                selected = solve_classical(num_vertices, edges)
                reason = (
                    "QAOA exceeded the configured simulation limit; "
                    "used exact classical search instead."
                )
            else:
                selected = solve_max_cut(
                    num_vertices, edges, layers=1, seed=seed
                )
                reason = (
                    "Selected using ML-predicted runtime. "
                    "Predictions are estimates, not guarantees."
                )
        else:
            raise ValueError(
                f"Unknown solver selected by ML: {ml_choice}"
            )

        results = [selected]
        if predicted_times is not None and "reason" not in locals():
            reason = (
                "Selected using ML-predicted runtime. "
                "Predictions are estimates, not guarantees."
            )

    elif objective == "quality":
        if num_vertices > MAX_EXACT_VERTICES:
            raise ValueError(
                f"Exact quality mode is limited to "
                f"{MAX_EXACT_VERTICES} vertices."
            )
        selected = solve_classical(num_vertices, edges)
        results = [selected]
        reason = "Exact classical search guarantees the optimum."

    elif objective == "quantum":
        if num_vertices > MAX_QAOA_VERTICES:
            raise ValueError(
                f"QAOA simulation is limited to "
                f"{MAX_QAOA_VERTICES} vertices."
            )
        selected = solve_max_cut(
            num_vertices, edges, layers=1, seed=seed
        )
        results = [selected]
        reason = (
            "Selected for quantum-algorithm exploration. "
            "This uses a classical statevector simulation, "
            "not quantum hardware."
        )

    else:
        results = run_all_solvers(num_vertices, edges, seed)
        selected = None
        reason = "All three solvers are shown for comparison."

    return {
        "objective": objective,
        "recommendation": (
            selected["algorithm"] if selected else None
        ),
        "reason": reason,
        "predicted_times": predicted_times,
        "results": results,
    }

    possible_edges = num_vertices * (num_vertices - 1) / 2

    edge_density = (
        num_edges / possible_edges if possible_edges else 0.0
    )

    degrees = [0] * num_vertices

    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    average_degree = (
        sum(degrees) / num_vertices if num_vertices else 0.0
    )
    max_degree = max(degrees, default=0)

    return {
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "edge_density": edge_density,
        "average_degree": average_degree,
        "max_degree": max_degree,
    }


def run_all_solvers(num_vertices, edges, seed=42):
    edges = list(edges)

    return [
        solve_classical(num_vertices, edges),
        solve_heuristic(num_vertices, edges, seed=seed),
        solve_max_cut(num_vertices, edges, layers=1, seed=seed),
    ]



def recommend_solver(
    num_vertices,
    edges,
    objective="fastest",
    seed=42,
    models=None,
):
    valid_objectives = {"fastest", "quality", "quantum", "compare"}

    if objective not in valid_objectives:
        raise ValueError(
            f"Invalid objective: {objective}. "
            f"Choose from {sorted(valid_objectives)}."
        )

    edges = list(edges)
    predicted_times = None

    if objective == "fastest":
        if models is None:
            models, _ = train_runtime_models()

        features = get_graph_features(num_vertices, edges)

        ml_choice, predicted_times = ml_recommend_solver(
            models,
            **features,
        )

        if ml_choice == "classical_exact":
            selected = solve_classical(num_vertices, edges)
        elif ml_choice == "qaoa_simulator":
            selected = solve_max_cut(
                num_vertices, edges, layers=1, seed=seed
            )
        else:
            raise ValueError(f"Unknown solver selected by ML: {ml_choice}")

        results = [selected]

        reason = (
            "Selected using ML-predicted runtime. "
            "Only the selected solver was executed. "
            "Predictions are estimates, not guarantees."
        )

    elif objective == "quality":
        selected = solve_classical(num_vertices, edges)
        results = [selected]
        reason = "Exact classical search guarantees the optimum."

    elif objective == "quantum":
        selected = solve_max_cut(
            num_vertices, edges, layers=1, seed=seed
        )
        results = [selected]
        reason = (
            "Selected for quantum-algorithm exploration. "
            "This uses a classical statevector simulation, "
            "not quantum hardware."
        )

    else:
        results = run_all_solvers(num_vertices, edges, seed)
        selected = None
        reason = "All three solvers are shown for comparison."

    return {
        "objective": objective,
        "recommendation": (
            selected["algorithm"] if selected else None
        ),
        "reason": reason,
        "predicted_times": predicted_times,
        "results": results,
    }

def recommend_solver(
    num_vertices,
    edges,
    objective="fastest",
    seed=42,
    models=None,
):
    valid_objectives = {"fastest", "quality", "quantum", "compare"}

    if objective not in valid_objectives:
        raise ValueError(
            f"Invalid objective: {objective}. "
            f"Choose from {sorted(valid_objectives)}."
        )

    edges = list(edges)
    predicted_times = None

    if objective == "fastest":
        if models is None:
            models, _ = train_runtime_models()

        features = get_graph_features(num_vertices, edges)

        ml_choice, predicted_times = ml_recommend_solver(
            models,
            **features,
        )

        if ml_choice == "classical_exact":
            selected = solve_classical(num_vertices, edges)
        elif ml_choice == "qaoa_simulator":
            selected = solve_max_cut(
                num_vertices, edges, layers=1, seed=seed
            )
        else:
            raise ValueError(f"Unknown solver selected by ML: {ml_choice}")

        results = [selected]

        reason = (
            "Selected using ML-predicted runtime. "
            "Only the selected solver was executed. "
            "Predictions are estimates, not guarantees."
        )

    elif objective == "quality":
        selected = solve_classical(num_vertices, edges)
        results = [selected]
        reason = "Exact classical search guarantees the optimum."

    elif objective == "quantum":
        selected = solve_max_cut(
            num_vertices, edges, layers=1, seed=seed
        )
        results = [selected]
        reason = (
            "Selected for quantum-algorithm exploration. "
            "This uses a classical statevector simulation, "
            "not quantum hardware."
        )

    else:
        results = run_all_solvers(num_vertices, edges, seed)
        selected = None
        reason = "All three solvers are shown for comparison."

    return {
        "objective": objective,
        "recommendation": (
            selected["algorithm"] if selected else None
        ),
        "reason": reason,
        "predicted_times": predicted_times,
        "results": results,
    }

