"""Download and parse OpenStreetMap road networks.

Provides a single entry point: `load_city(city_name)` which returns a
(road_network.RoadNetwork, list of traffic lights) tuple.
"""

import logging
import os
import random
from typing import Optional

import osmnx as ox
from shapely.geometry import Point, LineString

import config
from src import road_network

log = logging.getLogger(__name__)

# ── Type aliases ─────────────────
NodeID = int
EdgeKey = int  # OSM uses 0 as default key for single edges


def _normalize_city_name(name: str) -> str:
    """Strip whitespace and append country hint if missing."""
    name = name.strip()
    if "," not in name and name:
        name = f"{name}, USA"  # sensible default
    return name


def load_city(city_name: str) -> tuple[road_network.RoadNetwork, list["TrafficLight"]]:
    """Download OSM data for *city_name* and build a RoadNetwork + traffic lights.

    Args:
        city_name: Human-readable place name, e.g. "Manhattan, NY, USA".

    Returns:
        A (RoadNetwork, traffic_lights) tuple. Traffic lights are positioned
        at intersections where OSM tags indicate traffic_signals.

    Raises:
        ValueError: If the city cannot be found or has no drivable roads.
        TimeoutError: If the download times out.
    """
    city_name = _normalize_city_name(city_name)
    log.info("Loading city: %s", city_name)

    # Cache .graphml to avoid re-downloading
    cache_dir = config.OSM_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = city_name.replace(",", "_").replace(" ", "_")
    cache_path = os.path.join(cache_dir, f"{safe_name}.graphml")

    if os.path.exists(cache_path):
        log.info("Loading cached graph from %s", cache_path)
        graph = ox.load_graphml(cache_path)
    else:
        try:
            graph = ox.graph_from_place(
                city_name,
                network_type="drive",
                simplify=True,
                retain_all=False,
                truncate_by_edge=False,
            )
            ox.save_graphml(graph, cache_path)
            log.info("Saved graph to cache: %s", cache_path)
        except Exception as exc:
            raise ValueError(
                f"Could not load city '{city_name}'. Check spelling or network."
            ) from exc

    # Convert from MultiDiGraph to our RoadNetwork
    rn = road_network.RoadNetwork.from_osmnx(graph)

    # Extract traffic-light nodes
    traffic_lights: list = []  # we'll import TrafficLight later to avoid circular
    from src.traffic_light import TrafficLight

    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("highway") == "traffic_signals":
            x, y = node_data.get("x", 0.0), node_data.get("y", 0.0)
            # Convert coordinates to screen space later; store raw lon/lat for now
            traffic_lights.append(
                TrafficLight(
                    node_id=node_id,
                    position=(x, y),
                    green_seconds=config.TRAFFIC_LIGHT_GREEN_SECONDS,
                    yellow_seconds=config.TRAFFIC_LIGHT_YELLOW_SECONDS,
                    red_seconds=config.TRAFFIC_LIGHT_RED_SECONDS,
                    timer=random.random() * (config.TRAFFIC_LIGHT_GREEN_SECONDS
                                            + config.TRAFFIC_LIGHT_YELLOW_SECONDS
                                            + config.TRAFFIC_LIGHT_RED_SECONDS),
                )
            )

    log.info(
        "Loaded %d nodes, %d edges, %d traffic lights",
        rn.node_count,
        rn.edge_count,
        len(traffic_lights),
    )
    return rn, traffic_lights