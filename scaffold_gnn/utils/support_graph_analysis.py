import networkx as nx

from scaffold_gnn.sparsifiers.support_graph import SupportGraphSparsifier


def trace_support_graph(
    G,
    delta=0.4,
    init_support="mst",
    node_congestion=True,
    sampling_mode="full",
    sample_size=1000,
    seed=None,
):
    """Trace support-graph construction step by step.

    Returns a dict with:
    - ``base_support``: initial support graph
    - ``final_support``: final support graph after augmentation
    - ``history``: list of per-iteration snapshots
    - ``target_edges``: stopping threshold ``delta * m``
    """
    sparsifier = SupportGraphSparsifier(
        delta=delta,
        init_support=init_support,
        node_congestion=node_congestion,
        sampling_mode=sampling_mode,
        sample_size=sample_size,
        seed=seed,
    )
    G = sparsifier._ensure_nx(G)

    if init_support == "mst":
        H = nx.minimum_spanning_tree(G, weight=None, algorithm="kruskal")
    elif init_support == "spanner":
        raise NotImplementedError(
            "Tracing currently supports init_support='mst' only."
        )
    else:
        raise ValueError(f"Unknown init_support: {init_support}")

    base_support = H.copy()
    m = G.number_of_edges()
    target_edges = delta * m
    history = []

    while H.number_of_edges() < target_edges:
        missing_edges = set(G.edges()) - set(H.edges())
        if not missing_edges:
            break

        sampled_edges = sparsifier._sample_edges(G, H, missing_edges)
        congestions = {}
        disconnected_edges = []
        max_congestion = 0
        max_congestion_obj = None
        max_dilation = 0

        for edge in sampled_edges:
            try:
                path = nx.shortest_path(H, source=edge[0], target=edge[1], weight=None)
            except nx.NetworkXNoPath:
                disconnected_edges.append(edge)
                continue

            dilation = len(path) - 1
            max_dilation = max(max_dilation, dilation)

            if node_congestion:
                path_items = path
            else:
                path_items = [
                    (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                    for i in range(len(path) - 1)
                ]

            for item in path_items:
                if item not in congestions:
                    congestions[item] = {"count": 0, "edges": []}
                congestions[item]["count"] += 1
                congestions[item]["edges"].append({"edge": edge, "dilation": dilation, "path": path})
                if congestions[item]["count"] > max_congestion:
                    max_congestion = congestions[item]["count"]
                    max_congestion_obj = item

        iteration = {
            "support_edges_before": sorted(tuple(sorted(edge)) for edge in H.edges()),
            "sampled_edges": sampled_edges,
            "disconnected_edges": disconnected_edges,
            "max_congestion": max_congestion,
            "max_congestion_obj": max_congestion_obj,
            "max_dilation": max_dilation,
            "edge_routes": [],
            "top_candidates": [],
            "chosen_edge": None,
        }

        for edge in sampled_edges:
            route = {"edge": edge, "path": None, "dilation": None, "touches_bottleneck": False}
            try:
                path = nx.shortest_path(H, source=edge[0], target=edge[1], weight=None)
                dilation = len(path) - 1
                route["path"] = path
                route["dilation"] = dilation
                if max_congestion_obj is not None:
                    if node_congestion:
                        route["touches_bottleneck"] = max_congestion_obj in path
                    else:
                        bottleneck_edge = max_congestion_obj
                        path_edges = [
                            (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                            for i in range(len(path) - 1)
                        ]
                        route["touches_bottleneck"] = bottleneck_edge in path_edges
            except nx.NetworkXNoPath:
                pass
            iteration["edge_routes"].append(route)

        iteration["edge_routes"] = sorted(
            iteration["edge_routes"],
            key=lambda item: (
                item["dilation"] is None,
                -(item["dilation"] or -1),
                item["edge"],
            ),
        )

        if not congestions:
            if disconnected_edges:
                chosen_edge = disconnected_edges[0]
                iteration["chosen_edge"] = chosen_edge
                H.add_edge(*chosen_edge)
                iteration["support_edges_after"] = sorted(tuple(sorted(edge)) for edge in H.edges())
                history.append(iteration)
                continue
            break

        ranked_candidates = sorted(
            congestions[max_congestion_obj]["edges"],
            key=lambda item: (item["dilation"], item["edge"]),
            reverse=True,
        )
        iteration["top_candidates"] = ranked_candidates[:10]
        chosen_edge = ranked_candidates[0]["edge"]
        iteration["chosen_edge"] = chosen_edge

        H.add_edge(*chosen_edge)
        iteration["support_edges_after"] = sorted(tuple(sorted(edge)) for edge in H.edges())
        history.append(iteration)

    return {
        "base_support": base_support,
        "final_support": H.copy(),
        "history": history,
        "target_edges": target_edges,
    }


def initial_support_graph(G, init_support="mst"):
    """Return the initial support graph before augmentation."""
    if init_support != "mst":
        raise NotImplementedError("Only init_support='mst' is supported here.")
    return nx.minimum_spanning_tree(G, weight=None, algorithm="kruskal")
