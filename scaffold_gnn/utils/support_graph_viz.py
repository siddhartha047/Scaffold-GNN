import io

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image


def support_graph_at_step(base_graph, added_edges, step):
    H = base_graph.copy()
    for edge in added_edges[:step]:
        H.add_edge(*edge)
    return H


def draw_support_progression_frame(
    G,
    pos,
    base_support,
    added_edges,
    step,
    title=None,
    subtitle=None,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    H = support_graph_at_step(base_support, added_edges, step)

    base_edges = sorted(tuple(sorted(edge)) for edge in base_support.edges())
    previous_added = [tuple(sorted(edge)) for edge in added_edges[: max(0, step - 1)]]
    newest_added = [tuple(sorted(added_edges[step - 1]))] if step > 0 else []
    missing_edges = [
        tuple(sorted(edge))
        for edge in G.edges()
        if not H.has_edge(*edge)
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=missing_edges,
        width=0.8,
        alpha=0.15,
        edge_color="#808080",
    )
    nx.draw_networkx_edges(
        H,
        pos,
        ax=ax,
        edgelist=base_edges,
        width=1.3,
        alpha=0.85,
        edge_color="#9e9e9e",
    )
    if previous_added:
        nx.draw_networkx_edges(
            H,
            pos,
            ax=ax,
            edgelist=previous_added,
            width=2.8,
            edge_color="#ffb347",
        )
    if newest_added:
        nx.draw_networkx_edges(
            H,
            pos,
            ax=ax,
            edgelist=newest_added,
            width=3.8,
            edge_color="#d62728",
        )

    nx.draw_networkx_nodes(
        H,
        pos,
        ax=ax,
        node_color="#d9edf7",
        node_size=240,
        edgecolors="black",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=8)

    if title is None:
        if step == 0:
            title = f"Step 0: MST support ({H.number_of_edges()} edges)"
        else:
            title = f"Step {step}: add edge {added_edges[step - 1]} ({H.number_of_edges()} edges)"
    ax.set_title(title, fontsize=12)

    if subtitle:
        ax.text(
            0.5,
            0.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_to_pil_image(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    return image.copy()


def save_support_progression_gif(
    output_path,
    G,
    pos,
    base_support,
    added_edges,
    durations_ms=None,
):
    if durations_ms is None:
        durations_ms = [1200] + [900] * max(len(added_edges) - 1, 0) + [1600]

    frames = []
    total_steps = len(added_edges)
    for step in range(total_steps + 1):
        subtitle = (
            "Gray = MST edges, orange = previously added edges, red = newest edge"
        )
        fig = draw_support_progression_frame(
            G,
            pos,
            base_support,
            added_edges,
            step,
            subtitle=subtitle,
        )
        frames.append(fig_to_pil_image(fig))

    if len(durations_ms) != len(frames):
        raise ValueError("durations_ms must match the number of frames")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0,
        disposal=2,
    )
