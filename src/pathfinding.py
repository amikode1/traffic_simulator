"""Pathfinding algorithms for the road network.

Provides a unified interface for Dijkstra, A*, BFS, and Yen's K-Shortest
Paths. All functions accept a working graph (blocked edges removed) and
return a list of node IDs representing the path.
"""

import heapq
import logging
from collections import deque
from typing import Callable, Optional

import networkx as nx

import config
from src.road_network import NodeID

log = logging.getLogger(__name__)

# ── Type aliases ─────────────────
HeuristicFunc = Callable[[NodeID, NodeID], float]


# ── Main dispatcher ──────────────

def find_path(
    graph: nx.MultiDiGraph,
    origin: NodeID,
    destination: NodeID,
    algorithm: str = "dijkstra",
    k: int = 3,
    car_counts: Optional[dict[tuple[NodeID, NodeID, int], int]] = None,
) -> Optional[list[NodeID] | list[list[NodeID]]]:
    """Find a path between two nodes using the specified algorithm.

    Args:
        graph: The working graph (blocked edges already removed).
        origin: Start node ID.
        destination: End node ID.
        algorithm: One of 'dijkstra', 'a_star', 'bfs', 'yen_k_shortest', 'selfish'.
        k: Number of alternative routes for Yen's algorithm.
        car_counts: Map of (u, v, key) -> car count, used by the selfish algorithm.

    Returns:
        A list of node IDs (the path), or a list of paths for Yen's algorithm,
        or None if no path exists.

    Raises:
        ValueError: If the algorithm name is unknown.
    """
    if origin == destination:
        return [origin]

    algorithm = algorithm.lower().replace("-", "_").replace(" ", "_")

    if algorithm == "dijkstra":
        return _dijkstra(graph, origin, destination)
    elif algorithm == "a_star":
        return _a_star(graph, origin, destination)
    elif algorithm == "bfs":
        return _bfs(graph, origin, destination)
    elif algorithm in ("yen_k_shortest", "yen", "k_shortest"):
        return _yen_k_shortest(graph, origin, destination, k=k)
    elif algorithm == "selfish":
        return _selfish(graph, origin, destination, car_counts or {})
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Available: {config.AVAILABLE_ALGORITHMS}"
        )


# ── Dijkstra ─────────────────────

def _dijkstra(
    graph: nx.MultiDiGraph,
    origin: NodeID,
    destination: NodeID,
) -> Optional[list[NodeID]]:
    """Shortest path using Dijkstra's algorithm (weighted by road length)."""
    try:
        path = nx.shortest_path(
            graph, source=origin, target=destination, weight="length_meters"
        )
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ── A* ───────────────────────────

def _a_star(
    graph: nx.MultiDiGraph,
    origin: NodeID,
    destination: NodeID,
) -> Optional[list[NodeID]]:
    """Shortest path using A* with a Euclidean distance heuristic."""
    # Build a simple Euclidean heuristic from node positions
    pos = {n: (d.get("x", 0.0), d.get("y", 0.0)) for n, d in graph.nodes(data=True)}

    def heuristic(a: NodeID, b: NodeID) -> float:
        if a in pos and b in pos:
            # Approx: 1 degree longitude ~ 111 000 m at equator
            # Crude but sufficient for heuristic
            dx = (pos[a][0] - pos[b][0]) * 111_000.0
            dy = (pos[a][1] - pos[b][1]) * 111_000.0
            return (dx * dx + dy * dy) ** 0.5
        return 0.0

    try:
        path = nx.astar_path(
            graph,
            source=origin,
            target=destination,
            heuristic=heuristic,
            weight="length_meters",
        )
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ── BFS (unweighted) ─────────────

def _bfs(
    graph: nx.MultiDiGraph,
    origin: NodeID,
    destination: NodeID,
) -> Optional[list[NodeID]]:
    """Unweighted shortest path using BFS (ignores road length)."""
    try:
        path = nx.shortest_path(graph, source=origin, target=destination)
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ── Yen's K-Shortest Paths ───────

def _yen_k_shortest(
    graph: nx.MultiDiGraph,
    origin: NodeID,
    destination: NodeID,
    k: int = 3,
) -> Optional[list[list[NodeID]]]:
    """Return up to *k* shortest paths between origin and destination.

    Converts the MultiDiGraph to a simple DiGraph (keeping the shortest
    edge per pair) for compatibility with NetworkX's Yen's algorithm.
    """
    from networkx.algorithms.simple_paths import shortest_simple_paths

    # Convert MultiDiGraph → DiGraph (keep shortest edge per pair)
    simple_graph = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        length = data.get("length_meters", 1.0)
        # Keep the edge with the smallest length
        if simple_graph.has_edge(u, v):
            existing = simple_graph[u][v].get("length_meters", float("inf"))
            if length < existing:
                simple_graph[u][v]["length_meters"] = length
        else:
            simple_graph.add_edge(u, v, length_meters=length)

    try:
        paths: list[list[NodeID]] = []
        for i, path in enumerate(
            shortest_simple_paths(simple_graph, origin, destination, weight="length_meters")
        ):
            if i >= k:
                break
            paths.append(path)
        return paths if paths else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def compute_path_length(
    graph: nx.MultiDiGraph, path: list[NodeID]
) -> float:
    """Compute the total length in metres of a node path."""
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = graph.get_edge_data(u, v)
        if edge_data:
            key = next(iter(edge_data))
            total += edge_data[key].get("length_meters", 0.0)
    return total


# ── BPR congestion function (selfish algorithm) ────────────

def bpr_travel_time(
    free_flow_time: float,
    car_count: int,
    lanes: int,
) -> float:
    """Compute the BPR (Bureau of Public Roads) travel time for a road segment.

    Args:
        free_flow_time: Time to traverse the road with no congestion (seconds).
        car_count: Current number of cars on this road segment.
        lanes: Number of lanes on this road segment.

    Returns:
        Effective travel time in seconds, inflated by congestion.
    """
    capacity = max(lanes, 1) * config.CAPACITY_PER_LANE
    v_c_ratio = car_count / capacity
    congestion = 1.0 + config.BPR_ALPHA * (v_c_ratio ** config.BPR_BETA)
    return free_flow_time * congestion


def build_bpr_graph(
    graph: nx.MultiDiGraph,
    car_counts: dict[tuple[NodeID, NodeID, int], int],
) -> nx.DiGraph:
    """Pre-convert a MultiDiGraph into a simple DiGraph with BPR edge weights.

    This is the expensive part (iterates all edges, computes BPR costs).
    Call it ONCE per reroute cycle, then reuse the result for many
    individual pathfinding calls.

    Args:
        graph: The working graph (blocked edges removed).
        car_counts: Map of (u, v, key) -> current car count.

    Returns:
        A nx.DiGraph where every edge has a ``bpr_cost`` attribute.
    """
    simple = nx.DiGraph()
    for u, v, key, data in graph.edges(data=True, keys=True):
        free_flow_speed = data.get("speed_kph", config.DEFAULT_SPEED_KPH) / 3.6
        length_m = data.get("length_meters", 1.0)
        if free_flow_speed <= 0:
            free_flow_speed = 1.0
        free_flow_time = length_m / free_flow_speed
        lanes = data.get("lanes", 1)

        total_count = 0
        for (cu, cv, ck), cnt in car_counts.items():
            if cu == u and cv == v:
                total_count += cnt

        bpr_cost = bpr_travel_time(free_flow_time, total_count, lanes)

        if simple.has_edge(u, v):
            existing = simple[u][v].get("bpr_cost", float("inf"))
            if bpr_cost < existing:
                simple[u][v]["bpr_cost"] = bpr_cost
        else:
            simple.add_edge(u, v, bpr_cost=bpr_cost)

    return simple


def find_on_bpr_graph(
    bpr_graph: nx.DiGraph,
    origin: NodeID,
    destination: NodeID,
) -> Optional[list[NodeID]]:
    """Shortest path on a pre-built BPR-weighted DiGraph (fast, no conversion).

    Args:
        bpr_graph: A DiGraph previously built by :func:`build_bpr_graph`.
        origin: Start node ID.
        destination: End node ID.

    Returns:
        A list of node IDs (the path), or None if no path exists.
    """
    if origin == destination:
        return [origin]
    try:
        return nx.shortest_path(
            bpr_graph,
            source=origin,
            target=destination,
            weight="bpr_cost",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _selfish(
    graph: nx.MultiDiGraph,
    origin: NodeID,
    destination: NodeID,
    car_counts: dict[tuple[NodeID, NodeID, int], int],
) -> Optional[list[NodeID]]:
    """Shortest path using BPR congestion costs ("selfish" routing).

    Convenience wrapper — builds the BPR graph from scratch, then finds
    the path.  For bulk rerouting, use :func:`build_bpr_graph` once and
    :func:`find_on_bpr_graph` many times.

    Args:
        graph: The working graph (blocked edges removed).
        origin: Start node ID.
        destination: End node ID.
        car_counts: Map of (u, v, key) -> current car count.

    Returns:
        A list of node IDs (the path), or None if no path exists.
    """
    bpr_graph = build_bpr_graph(graph, car_counts)
    return find_on_bpr_graph(bpr_graph, origin, destination)