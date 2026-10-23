from graph_visualizer import create_graph_image
from flask import Flask, render_template, request
from decision_engine import recommend_solver, train_runtime_models

app = Flask(__name__)

print("Training ML runtime models...")
MODELS, _ = train_runtime_models()
print("Models ready!")


def parse_graph(num_vertices, raw_edges):
    if not 2 <= num_vertices <= 100:
        raise ValueError(
            "Number of vertices must be between 2 and 100."
        )

    if not raw_edges.strip():
        raise ValueError(
            "Please enter at least one edge, e.g. 0-1, 1-2."
        )

    edges = []

    for item in raw_edges.split(","):
        item = item.strip()
        parts = item.split("-")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid edge '{item}'. Use the format 0-1."
            )

        try:
            u, v = map(int, parts)
        except ValueError:
            raise ValueError(
                f"Invalid edge '{item}'. Vertex labels must be integers."
            )

        if not (0 <= u < num_vertices and 0 <= v < num_vertices):
            raise ValueError(
                f"Edge '{item}' uses a vertex outside the valid range "
                f"0 to {num_vertices - 1}."
            )

        if u == v:
            raise ValueError(
                f"Edge '{item}' is a self-loop. Connect two different vertices."
            )

        edge = (min(u, v), max(u, v))

        if edge in edges:
            raise ValueError(
                f"Edge '{u}-{v}' is duplicated. Enter each edge only once."
            )

        edges.append(edge)

    return edges


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            num_vertices = int(request.form["num_vertices"])
            raw_edges = request.form["edges"]

            edges = parse_graph(num_vertices, raw_edges)
            objective = request.form["objective"]

            result = recommend_solver(
                num_vertices=num_vertices,
                edges=edges,
                objective=objective,
                models=MODELS,
            )

             # Generate graph visualization
            selected_result = next(
                (
                    solver for solver in result["results"]
                    if solver["algorithm"] == result["recommendation"]
                ),
                None,
            )

            if selected_result is None and result["results"]:
                selected_result = result["results"][0]

            result["graph_image"] = create_graph_image(
                num_vertices,
                edges,
                selected_result["partition"],
            ) if selected_result else None

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Unexpected solver error")
            error = (
                "Something went wrong while analyzing the graph. "
                "Please check your inputs and try again."
            )

    return render_template(
        "index.html",
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
