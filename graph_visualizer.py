
import io
import base64

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx


def create_graph_image(num_vertices, edges, partition):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    graph.add_edges_from(edges)

    # Consistent layout for the same graph
    positions = nx.spring_layout(graph, seed=42)

    # Separate vertices into the two Max-Cut partitions
    colors = [
        "#818cf8" if partition[node] == 0 else "#34d399"
        for node in graph.nodes()
    ]

    # Highlight edges crossing between the partitions
    edge_colors = [
        "#f59e0b" if partition[u] != partition[v] else "#64748b"
        for u, v in graph.edges()
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")

    nx.draw_networkx(
        graph,
        positions,
        ax=ax,
        with_labels=True,
        node_color=colors,
        edge_color=edge_colors,
        node_size=1000,
        font_color="white",
        font_weight="bold",
        width=2.5,
        font_size=12,
    )

    ax.set_title(
        "Max-Cut Graph",
        color="white",
        fontsize=16,
        pad=16,
    )
    ax.axis("off")

    legend_items = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="#818cf8",
            markersize=12,
            label="Partition 0",
        ),
        plt.Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="#34d399",
            markersize=12,
            label="Partition 1",
        ),
        plt.Line2D(
            [0], [0],
            color="#f59e0b",
            linewidth=3,
            label="Cut edge",
        ),
    ]

    legend = ax.legend(
        handles=legend_items,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        facecolor="#151c30",
        edgecolor="#303a55",
        fontsize=9,
        borderpad=1.2,
        labelspacing=1.2,
    )

    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()

    image_buffer = io.BytesIO()
    fig.savefig(
        image_buffer,
        format="png",
        dpi=120,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    image_buffer.seek(0)
    encoded_image = base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")

    return encoded_image
