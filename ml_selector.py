
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error


DATA_FILE = "benchmark_results.csv"

FEATURES = [
    "num_vertices",
    "num_edges",
    "edge_density",
    "average_degree",
    "max_degree",
]


def train_runtime_models():
    data = pd.read_csv(DATA_FILE)

    # Keep the two solvers separate.
    classical = data[data["solver"] == "classical_exact"].copy()
    qaoa = data[data["solver"] == "qaoa_simulator"].copy()

    # Split by graph ID to prevent the same graph appearing
    # in both the training and testing sets.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(classical, groups=classical["graph_id"])
    )

    models = {}
    results = {}

    for name, solver_data in [
        ("classical_exact", classical),
        ("qaoa_simulator", qaoa),
    ]:
        train = solver_data.iloc[train_idx]
        test = solver_data.iloc[test_idx]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        )

        # Predict the logarithm of runtime because the two
        # solvers operate on very different time scales.
        model.fit(
            train[FEATURES],
            train["runtime_seconds"].clip(lower=1e-9).apply(
                __import__("numpy").log
            ),
        )

        predictions = __import__("numpy").exp(
            model.predict(test[FEATURES])
        )

        actual = test["runtime_seconds"].to_numpy()

        models[name] = model
        results[name] = {
            "actual": actual,
            "predicted": predictions,
            "mae": mean_absolute_error(actual, predictions),
        }

    return models, results


def recommend_solver(models, num_vertices, num_edges,
                     edge_density, average_degree, max_degree):
    graph = pd.DataFrame([{
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "edge_density": edge_density,
        "average_degree": average_degree,
        "max_degree": max_degree,
    }])

    predicted_times = {}

    for name, model in models.items():
        predicted_log_time = model.predict(graph[FEATURES])[0]
        predicted_times[name] = float(
            __import__("numpy").exp(predicted_log_time)
        )

    recommendation = min(predicted_times, key=predicted_times.get)

    return recommendation, predicted_times


if __name__ == "__main__":
    models, results = train_runtime_models()

    print("=== Runtime Prediction Evaluation ===")

    for name, result in results.items():
        print(f"\nSolver: {name}")
        print(f"Test MAE: {result['mae']:.6f} seconds")

    # Example: a graph with 4 vertices and 5 edges.
    recommendation, predicted_times = recommend_solver(
        models,
        num_vertices=4,
        num_edges=5,
        edge_density=5 / 6,
        average_degree=2.5,
        max_degree=3,
    )

    print("\n=== Solver Recommendation ===")
    for solver, runtime in predicted_times.items():
        print(f"{solver}: predicted {runtime:.6f} seconds")

    print(f"\nRecommended solver: {recommendation}")
