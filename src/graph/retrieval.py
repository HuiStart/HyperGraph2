"""
Graph retrieval utilities for finding related evidence.
"""

from typing import Any

import networkx as nx

from src.graph.builder import HyperGraphBuilder


def find_related_evidence(
    builder: HyperGraphBuilder,
    dimension: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Find top-k evidence related to a dimension.

    Uses both hyperedge connections and graph shortest paths.
    """
    dim_node = f"dimension:{dimension.lower()}"
    if dim_node not in builder.graph:
        return []

    # Get directly connected evidence via hyperedges
    direct = builder.get_related_evidence(dimension)

    # Get evidence reachable via graph paths
    path_scores = {}
    for node in builder.graph.nodes():
        if node.startswith("evidence:"):
            try:
                path_length = nx.shortest_path_length(
                    builder.graph, source=dim_node, target=node, weight="weight"
                )
                path_scores[node] = 1.0 / (1.0 + path_length)
            except nx.NetworkXNoPath:
                continue

    # Combine and rank
    seen = set()
    ranked = []
    for ev in direct:
        node_id = ev.get("text", "")[:50]
        if node_id not in seen:
            seen.add(node_id)
            ranked.append((ev, ev.get("hyperedge_weight", 0.5)))

    for node, score in sorted(path_scores.items(), key=lambda x: -x[1]):
        if node in builder.graph:
            data = dict(builder.graph.nodes[node])
            node_id = data.get("text", "")[:50]
            if node_id not in seen:
                seen.add(node_id)
                ranked.append((data, score))

    # Sort by score and return top_k
    ranked.sort(key=lambda x: -x[1])
    return [item[0] for item in ranked[:top_k]]


def get_dimension_subgraph(
    builder: HyperGraphBuilder,
    dimension: str,
) -> nx.Graph:
    """Extract subgraph containing nodes related to a dimension."""
    dim_node = f"dimension:{dimension.lower()}"
    if dim_node not in builder.graph:
        return nx.Graph()

    # BFS to find connected nodes
    nodes = set([dim_node])
    for edge in builder.hyperedges:
        if dim_node in edge["nodes"]:
            nodes.update(edge["nodes"])

    return builder.graph.subgraph([n for n in nodes if n in builder.graph]).copy()
