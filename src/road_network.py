"""Directed road network wrapper around NetworkX.

Encapsulates a NetworkX MultiDiGraph with lane counts, directions,
speed limits, and geometry for each road segment.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

import config

log = logging.getLogger(__name__)

NodeID = int
EdgeKey = int  # OSM edge key (usually 0 for single carriageway)


@dataclass(frozen=True, order=True)
class RoadEdge:
    """A directed road segment between two intersections.

    Attributes:
        u: Source node ID.
        v: Target node ID.
        key: Edge key (allows parallel edges).
        lanes: Number of lanes in this direction.
        length_meters: Road length in metres.
        speed_kph: Speed limit in km/h.
        geometry: Shapely LineString of the road curve.
        oneway: Whether this is a one-way street.
        name: Street name (if available).
    """

    u: NodeID
    v: NodeID
    key: int
    lanes: int = 1
    length_meters: float = 0.0
    speed_kph: float = config.DEFAULT_SPEED_KPH
    geometry: Optional[LineString] = None
    oneway: bool = True
    name: str = ""


@dataclass
class RoadNetwork:
    """Directed graph of the city's drivable roads.

    Wraps a NetworkX MultiDiGraph. Nodes are intersections; edges are road
    segments. Provides methods for querying lanes, blocking, and pathfinding.
    """

    _graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    _blocked_edges: set[tuple[NodeID, NodeID, EdgeKey]] = field(default_factory=set)

    # ── Properties ──────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def blocked_edges(self) -> set[tuple[NodeID, NodeID, EdgeKey]]:
        """Return a copy of the blocked edges set."""
        return set(self._blocked_edges)

    # ── Construction ────────────────────────────────────────────

    @classmethod
    def from_osmnx(cls, graph: nx.MultiDiGraph) -> "RoadNetwork":
        """Build a RoadNetwork from an OSMnx MultiDiGraph.

        Extracts lane counts, speed limits, geometry, and street names.
        Missing attributes are filled with sensible defaults.
        """
        rn = cls()
        rn._graph = graph

        # Ensure all edges have the required attributes
        for u, v, key, data in graph.edges(data=True, keys=True):
            # Lanes: OSM may store as string, int, or None
            lanes_raw = data.get("lanes", 1)
            if lanes_raw is None:
                lanes_raw = 1
            if isinstance(lanes_raw, str):
                # Sometimes "2;3" means 2 forward + 3 backward — use first value
                lanes_raw = lanes_raw.split(";")[0]
            try:
                lanes = int(lanes_raw)
            except (ValueError, TypeError):
                lanes = 1
            lanes = max(lanes, 1)

            # Speed limit
            speed_raw = data.get("maxspeed", config.DEFAULT_SPEED_KPH)
            if isinstance(speed_raw, list):
                # OSM sometimes stores multiple speed limits (e.g., different directions)
                speed_raw = speed_raw[0] if speed_raw else config.DEFAULT_SPEED_KPH
            if isinstance(speed_raw, dict):
                # OSM may store maxspeed:forward / maxspeed:backward as a dict
                speed_raw = next(iter(speed_raw.values())) if speed_raw else config.DEFAULT_SPEED_KPH
            if isinstance(speed_raw, str):
                # "50 mph" or "50" -> convert to kph
                speed_raw = speed_raw.replace(" mph", "").strip()
                try:
                    speed_kph = float(speed_raw)
                except (ValueError, TypeError):
                    speed_kph = config.DEFAULT_SPEED_KPH
            else:
                speed_kph = float(speed_raw) if speed_raw else config.DEFAULT_SPEED_KPH

            # Geometry
            geometry = data.get("geometry")
            length_meters = data.get("length", 0.0)

            # Oneway: OSM can store -1 (reverse direction), "yes", "no", True, False
            oneway = data.get("oneway", True)
            if isinstance(oneway, int):
                oneway = oneway == 1  # -1 means reverse oneway (we keep as is for the edge direction)
            elif isinstance(oneway, str):
                oneway = oneway.lower() == "yes" or oneway == "1"

            name = data.get("name", "")

            rn._graph[u][v][key].update(
                {
                    "key": key,
                    "lanes": lanes,
                    "speed_kph": speed_kph,
                    "length_meters": length_meters,
                    "geometry": geometry,
                    "oneway": oneway,
                    "name": name,
                }
            )

        return rn

    # ── Edge queries ────────────────────────────────────────────

    def get_edge(self, u: NodeID, v: NodeID, key: EdgeKey = 0) -> Optional[dict[str, Any]]:
        """Return the data dict for an edge, or None if it doesn't exist."""
        try:
            return self._graph[u][v][key]
        except (KeyError, nx.NetworkXError):
            return None

    def get_edge_attributes(self, u: NodeID, v: NodeID, key: EdgeKey = 0) -> RoadEdge:
        """Return a typed RoadEdge for the given edge."""
        data = self.get_edge(u, v, key)
        if data is None:
            raise KeyError(f"Edge ({u}, {v}, {key}) not found")
        return RoadEdge(
            u=u,
            v=v,
            key=key,
            lanes=data.get("lanes", 1),
            length_meters=data.get("length_meters", 0.0),
            speed_kph=data.get("speed_kph", config.DEFAULT_SPEED_KPH),
            geometry=data.get("geometry"),
            oneway=data.get("oneway", True),
            name=data.get("name", ""),
        )

    def edges_from_node(self, u: NodeID) -> list[tuple[NodeID, NodeID, EdgeKey]]:
        """Return all outgoing edges from node *u*."""
        return list(self._graph.out_edges(u, keys=True))

    def edges_to_node(self, v: NodeID) -> list[tuple[NodeID, NodeID, EdgeKey]]:
        """Return all incoming edges to node *v*."""
        return list(self._graph.in_edges(v, keys=True))

    def is_edge_blocked(self, u: NodeID, v: NodeID, key: EdgeKey = 0) -> bool:
        """Check if an edge is currently blocked."""
        return (u, v, key) in self._blocked_edges

    def get_node_position(self, node_id: NodeID) -> Optional[tuple[float, float]]:
        """Return (lon, lat) for a node, or None if not found."""
        data = self._graph.nodes.get(node_id)
        if data is None:
            return None
        return (data.get("x", 0.0), data.get("y", 0.0))

    def get_all_nodes(self) -> list[NodeID]:
        """Return all node IDs in the graph."""
        return list(self._graph.nodes)

    def get_all_edges(self) -> list[tuple[NodeID, NodeID, EdgeKey]]:
        """Return all (u, v, key) tuples in the graph."""
        return list(self._graph.edges(keys=True))

    # ── Blocking / unblocking ──────────────────────────────────

    def block_edge(self, u: NodeID, v: NodeID, key: EdgeKey = 0) -> bool:
        """Block a road edge. Returns True if it was newly blocked."""
        if (u, v, key) not in self._graph.edges(keys=True):
            return False
        self._blocked_edges.add((u, v, key))
        log.debug("Blocked edge (%d, %d, %d)", u, v, key)
        return True

    def unblock_edge(self, u: NodeID, v: NodeID, key: EdgeKey = 0) -> bool:
        """Unblock a road edge. Returns True if it was previously blocked."""
        try:
            self._blocked_edges.remove((u, v, key))
            log.debug("Unblocked edge (%d, %d, %d)", u, v, key)
            return True
        except KeyError:
            return False

    def toggle_block_edge(self, u: NodeID, v: NodeID, key: EdgeKey = 0) -> bool:
        """Toggle blocked status. Returns the new blocked state."""
        if self.is_edge_blocked(u, v, key):
            self.unblock_edge(u, v, key)
            return False
        else:
            self.block_edge(u, v, key)
            return True

    # ── Pathfinding helpers ────────────────────────────────────

    def get_working_graph(self) -> nx.MultiDiGraph:
        """Return a copy of the graph with blocked edges removed.

        Pathfinding should use this to avoid blocked roads.
        """
        g = self._graph.copy()
        for u, v, key in self._blocked_edges:
            if g.has_edge(u, v, key):
                g.remove_edge(u, v, key)
        return g

    def get_random_node(self) -> Optional[NodeID]:
        """Return a random node for spawning cars."""
        import random
        nodes = list(self._graph.nodes)
        if not nodes:
            return None
        return random.choice(nodes)

    def get_random_edge(self) -> Optional[tuple[NodeID, NodeID, EdgeKey]]:
        """Return a random edge for spawning cars."""
        import random
        edges = self.get_all_edges()
        if not edges:
            return None
        return random.choice(edges)

    def get_node_degree(self, node_id: NodeID) -> int:
        """Return the number of outgoing edges from a node."""
        return self._graph.out_degree(node_id)