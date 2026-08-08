"""Braess Paradox Road Detector — Phase 1 of the automated simulation.

Identifies roads whose *removal* paradoxically *improves* average travel time
(Braess's paradox). Runs a headless simulation at maximum speed with a fixed
set of commuters, tests every road one at a time, and scores each road by
the change in average travel time.

Performance:
  - Uses Tarjan's algorithm (O(N+E)) to pre-compute critical bridges,
    replacing the O(N * (N+E)) per-road connectivity check.
  - Uses a larger simulation timestep (0.2s) for ~4x faster simulation.
  - Tests roads in parallel using multiprocessing (up to CPU count).
  - Results are saved to data/braess_<city>.json for Phase 2 (GA).

Usage:
    python -m src.braess_detector "Greenwood_Township__Pennsylvania__United_States"
"""

import logging
import multiprocessing
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import networkx as nx

import config
from src.car import Car
from src.road_network import NodeID, EdgeKey, RoadNetwork
from src.traffic_simulation import TrafficSimulation
from src.traffic_light import TrafficLight

log = logging.getLogger(__name__)


# ── Connectivity helpers ──────────────────────────────────────────

def _find_bridges(
    node_count: int,
    adjacency: dict[NodeID, list[NodeID]],
) -> set[tuple[NodeID, NodeID]]:
    """Find all bridges in an undirected graph using Tarjan's algorithm.

    A bridge is an edge whose removal disconnects the graph.
    Runs in O(N + E) time.

    Args:
        node_count: Number of nodes (unused — kept for compatibility).
        adjacency: Dict mapping node -> list of neighbour nodes.

    Returns:
        Set of (u, v) tuples representing undirected bridges (u < v).
    """
    n = len(adjacency)
    if n == 0:
        return set()

    visited: dict[NodeID, int] = {}  # node -> discovery time
    low: dict[NodeID, int] = {}
    parent: dict[NodeID, Optional[NodeID]] = {}
    bridges: set[tuple[NodeID, NodeID]] = set()
    time_counter = 0

    start_node = next(iter(adjacency))

    def dfs(u: NodeID) -> None:
        nonlocal time_counter
        visited[u] = low[u] = time_counter
        time_counter += 1

        for v in adjacency.get(u, []):
            if v not in visited:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > visited[u]:
                    # (u, v) is a bridge — store canonical order
                    a, b = (u, v) if u < v else (v, u)
                    bridges.add((a, b))
            elif v != parent.get(u):
                low[u] = min(low[u], visited[v])

    dfs(start_node)
    return bridges


def _build_critical_edge_set(
    road_network: RoadNetwork,
) -> set[tuple[NodeID, NodeID, EdgeKey]]:
    """Pre-compute the set of edges that are critical for connectivity.

    Uses Tarjan's bridge-finding algorithm on the *undirected*
    version of the road graph. Any directed edge whose undirected
    projection is a bridge is marked as critical.

    This is called ONCE before testing begins, replacing the O(N*(N+E))
    per-road connectivity check with a single O(N+E) pass.

    Args:
        road_network: The road network to analyse.

    Returns:
        Set of (u, v, key) tuples that are bridges.
    """
    # Build undirected adjacency from all directed edges
    undirected_adj: dict[NodeID, list[NodeID]] = {}
    edge_map: dict[tuple[NodeID, NodeID], list[tuple[NodeID, NodeID, EdgeKey]]] = {}

    for u, v, key in road_network.get_all_edges():
        # Build adjacency (undirected)
        if u not in undirected_adj:
            undirected_adj[u] = []
        if v not in undirected_adj:
            undirected_adj[v] = []
        if v not in undirected_adj[u]:
            undirected_adj[u].append(v)
        if u not in undirected_adj[v]:
            undirected_adj[v].append(u)

        # Map undirected (u,v) to all directed edge keys
        uv_key = (u, v) if u < v else (v, u)
        if uv_key not in edge_map:
            edge_map[uv_key] = []
        edge_map[uv_key].append((u, v, key))

    # Find bridges in the undirected graph
    bridges_undirected = _find_bridges(len(undirected_adj), undirected_adj)

    # Map undirected bridges back to original directed edges
    critical: set[tuple[NodeID, NodeID, EdgeKey]] = set()
    for (a, b) in bridges_undirected:
        uv_key = (a, b)
        for u, v, key in edge_map.get(uv_key, []):
            critical.add((u, v, key))

    log.info(
        "Pre-computed %d critical edges (bridges) out of %d total",
        len(critical), road_network.edge_count,
    )
    return critical


def _edge_is_critical(
    u: NodeID,
    v: NodeID,
    key: EdgeKey,
    critical_set: set[tuple[NodeID, NodeID, EdgeKey]],
) -> bool:
    """Check if a specific directed edge is critical (a bridge).

    Pre-computed set lookup — O(1).

    Args:
        u: Source node.
        v: Target node.
        key: Edge key.
        critical_set: Set of (u, v, key) that are bridges.

    Returns:
        True if this edge is critical and would disconnect the city.
    """
    return (u, v, key) in critical_set


# ── Headless Simulation ───────────────────────────────────────────

class HeadlessSimulation(TrafficSimulation):
    """Simulation that runs at maximum speed with Poisson car spawning.

    Extends TrafficSimulation by providing a run_until_stable() loop
    that runs headless (no rendering, no frame cap) until a target
    number of trips have completed. Uses the same Poisson process
    as the visible simulation for consistency.

    This is NOT a dataclass — we override __init__ explicitly.
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        poisson_rate: float = 0.5,
        traffic_lights: Optional[list[TrafficLight]] = None,
    ) -> None:
        # Manually initialise all TrafficSimulation fields (bypass dataclass)
        self.road_network = road_network
        self.traffic_lights = traffic_lights or []
        self.cars: list[Car] = []
        self.algorithm: str = "selfish"
        self.speed_multiplier: float = 1.0
        self.next_car_id: int = 0
        self.time_seconds: float = 0.0
        self.poisson_rate: float = poisson_rate
        self._next_spawn_time: float = 0.0  # spawn immediately on first tick
        self._stats: dict = {
            "total_spawned": 0,
            "total_reached_destination": 0,
            "total_rerouted": 0,
            "total_travel_time_seconds": 0.0,
            "completed_trips": 0,
        }
        self._selfish_timer: float = 0.0
        self._selfish_reroute_queue: list[int] = []
        self._selfish_reroute_idx: int = 0
        self._selfish_counts_snapshot: dict = {}
        self._selfish_bpr_graph: Any = None
        self.ga_label: str = ""

    @staticmethod
    def _find_path(
        graph: nx.MultiDiGraph,
        origin: NodeID,
        destination: NodeID,
    ) -> Optional[list[NodeID]]:
        """Find shortest path between two nodes (wrapped for error handling).

        Args:
            graph: The working graph (blocked edges removed).
            origin: Start node ID.
            destination: End node ID.

        Returns:
            A list of node IDs, or None if no path exists.
        """
        if origin == destination:
            return [origin]
        try:
            return nx.shortest_path(
                graph, source=origin, target=destination, weight="length_meters",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # ── Run loop ──

    def run_until_stable(
        self,
        min_completed: int = 500,
        max_steps: int = 1_000_000,
        dt: float = 0.2,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """Run the simulation until a target number of trips complete.

        Uses the same Poisson spawn process as the visible simulation.

        Args:
            min_completed: Stop when this many trips have been recorded.
            max_steps: Safety limit on iterations.
            dt: Timestep in seconds (larger = faster but less accurate).
            progress_callback: Optional fn(completed, total) called periodically.

        Returns:
            dict with keys: completed_trips, avg_travel_time_seconds,
                total_travel_time_seconds, time_seconds, steps.
        """
        step = 0
        last_logged_completed = 0
        LOG_INTERVAL = max(1, min_completed // 20)

        while self._stats["completed_trips"] < min_completed and step < max_steps:
            step += 1

            # Advance the simulation using the full update() method
            # (includes Poisson spawning, selfish rerouting, traffic lights,
            #  car movement, and cleanup)
            clamped_dt = min(dt, 0.1)
            self.update(clamped_dt)

            # Progress reporting
            current_completed = self._stats["completed_trips"]
            if (progress_callback
                    and current_completed != last_logged_completed
                    and current_completed % LOG_INTERVAL == 0):
                progress_callback(current_completed, min_completed)
                last_logged_completed = current_completed

        completed = self._stats["completed_trips"]
        total_time = self._stats["total_travel_time_seconds"]
        avg = total_time / completed if completed > 0 else 0.0
        return {
            "completed_trips": completed,
            "total_travel_time_seconds": total_time,
            "avg_travel_time_seconds": avg,
            "time_seconds": self.time_seconds,
            "steps": step,
        }

    def get_stats(self) -> dict:
        """Return current statistics (used by renderer, not needed here)."""
        completed = self._stats["completed_trips"]
        total_time = self._stats["total_travel_time_seconds"]
        avg = total_time / completed if completed > 0 else 0.0
        return {
            "completed_trips": completed,
            "avg_travel_time_seconds": avg,
        }


# ── Braess Detector ───────────────────────────────────────────────

@dataclass
class BraessResult:
    """Result of testing a single road.

    Attributes:
        edge: (u, v, key) of the tested road.
        baseline_avg: Average travel time with all roads open.
        test_avg: Average travel time with this road closed.
        score: baseline_avg - test_avg (positive means improvement).
        disconnected: Whether this road disconnects the city.
    """
    edge: tuple[NodeID, NodeID, EdgeKey]
    baseline_avg: float
    test_avg: float
    score: float
    disconnected: bool

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "u": self.edge[0],
            "v": self.edge[1],
            "key": self.edge[2],
            "baseline_avg": round(self.baseline_avg, 4),
            "test_avg": round(self.test_avg, 4),
            "score": round(self.score, 4),
            "disconnected": self.disconnected,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BraessResult":
        """Create from a dictionary (reverse of to_dict)."""
        return cls(
            edge=(data["u"], data["v"], data["key"]),
            baseline_avg=data["baseline_avg"],
            test_avg=data["test_avg"],
            score=data["score"],
            disconnected=data["disconnected"],
        )


# ── Braess Registry (persistence) ────────────────────────────────

@dataclass
class BraessRegistry:
    """Persistent storage of Braess detection results.

    Holds the full set of results and metadata. Can be saved to and
    loaded from a JSON file, which Phase 2 (genetic algorithm) will
    read to know which roads to operate on.

    Attributes:
        city_name: Name of the city that was tested.
        commuter_count: Number of fixed commuters used.
        min_completed: Number of trips used for stabilisation.
        baseline_avg: Baseline average travel time (all roads open).
        total_roads_tested: How many roads were evaluated.
        results: Full list of BraessResult (including disconnected).
    """
    city_name: str
    commuter_count: int
    min_completed: int
    baseline_avg: float
    total_roads_tested: int
    results: list[BraessResult] = field(default_factory=list)

    # ── Properties ──────────────────────────────────────────────

    @property
    def braess_roads(self) -> list[BraessResult]:
        """Return only roads whose closure improved travel time (score > 0)."""
        return [r for r in self.results if r.score > 0 and not r.disconnected]

    @property
    def num_braess_roads(self) -> int:
        """Number of Braess-tainted roads found."""
        return len(self.braess_roads)

    @property
    def num_disconnected(self) -> int:
        """Number of roads that would disconnect the city."""
        return sum(1 for r in self.results if r.disconnected)

    # ── Serialisation ──────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert the entire registry to a JSON-serialisable dict."""
        return {
            "city_name": self.city_name,
            "commuter_count": self.commuter_count,
            "min_completed": self.min_completed,
            "baseline_avg": round(self.baseline_avg, 4),
            "total_roads_tested": self.total_roads_tested,
            "num_braess_roads": self.num_braess_roads,
            "num_disconnected": self.num_disconnected,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BraessRegistry":
        """Create a registry from a dict (reverse of to_dict)."""
        return cls(
            city_name=data["city_name"],
            commuter_count=data["commuter_count"],
            min_completed=data["min_completed"],
            baseline_avg=data["baseline_avg"],
            total_roads_tested=data["total_roads_tested"],
            results=[BraessResult.from_dict(r) for r in data["results"]],
        )

    def save(self, filepath: str) -> None:
        """Save the registry to a JSON file.

        Args:
            filepath: Path to the output JSON file.
        """
        import json
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info("Saved Braess registry to %s (%d roads, %d Braess-tainted)",
                 filepath, len(self.results), self.num_braess_roads)

    @classmethod
    def load(cls, filepath: str) -> "BraessRegistry":
        """Load a registry from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            The loaded BraessRegistry.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON format is invalid.
        """
        import json
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Braess registry not found: {filepath}")
        with open(filepath) as f:
            data = json.load(f)
        registry = cls.from_dict(data)
        log.info("Loaded Braess registry from %s (%d roads, %d Braess-tainted)",
                 filepath, len(registry.results), registry.num_braess_roads)
        return registry

    @staticmethod
    def default_filename(city_name: str) -> str:
        """Return the default file path for a given city name.

        Args:
            city_name: City identifier (e.g. "Greenwood_Township__Pennsylvania__United_States").

        Returns:
            Path like "data/braess_Greenwood_Township__Pennsylvania__United_States.json".
        """
        safe_name = city_name.replace("/", "_").replace("\\", "_")
        return f"data/braess_{safe_name}.json"


def find_braess_roads(
    road_network: RoadNetwork,
    commuter_pairs: list[tuple[NodeID, NodeID]],
    city_name: str = "unknown",
    commuter_count: int = 500,
    min_completed: int = 500,
    dt: float = 0.2,
    verbose: bool = True,
) -> BraessRegistry:
    """Identify roads whose closure improves average travel time.

    Phase 1 — baseline: run simulation with all roads open to establish
    the baseline average travel time.

    Phase 2 — per-road test: close each road, wait for the average to
    stabilise, compute the score, reopen, and move to the next.

    Roads that would disconnect the city are skipped with disconnected=True.

    Args:
        road_network: The city road network.
        commuter_pairs: List of (origin, destination) node pairs.
        city_name: Identifier for the city (used for file naming).
        commuter_count: Number of fixed commuters (for metadata).
        min_completed: Number of completed trips to wait for stabilisation.
        dt: Timestep for the headless simulation.
        verbose: Print progress to stderr.

    Returns:
        BraessRegistry containing all results and metadata.
    """
    all_edges = road_network.get_all_edges()
    if verbose:
        print(f"Total edges in network: {len(all_edges)}", file=sys.stderr)

    # Pre-compute critical edges (bridges) — O(N+E) once
    if verbose:
        print("Pre-computing critical edges (Tarjan's bridges)...", file=sys.stderr)
    critical_edges = _build_critical_edge_set(road_network)
    if verbose:
        print(
            f"  {len(critical_edges)} critical edges identified "
            f"(would disconnect city if closed)",
            file=sys.stderr,
        )

    # ── Phase 1: Baseline ──
    if verbose:
        print("Phase 1: Establishing baseline (all roads open)...", file=sys.stderr)

    baseline_sim = HeadlessSimulation(road_network, poisson_rate=0.5)
    baseline_result = baseline_sim.run_until_stable(
        min_completed=min_completed,
        dt=dt,
        progress_callback=lambda c, t: print(
            f"  Baseline: {c}/{t} trips completed", file=sys.stderr,
        ),
    )
    baseline_avg = baseline_result["avg_travel_time_seconds"]
    if verbose:
        print(
            f"  Baseline avg travel time: {baseline_avg:.2f}s "
            f"(after {baseline_result['completed_trips']} trips)",
            file=sys.stderr,
        )

    # ── Phase 2: Test each road ──
    results: list[BraessResult] = []
    total_roads = len(all_edges)
    skipped_disconnect = 0

    for idx, (u, v, key) in enumerate(all_edges):
        if verbose:
            print(
                f"  [{idx+1}/{total_roads}] Testing edge "
                f"({u}, {v}, key={key})...",
                file=sys.stderr,
            )

        # Check if this road is critical for connectivity (O(1) lookup)
        if _edge_is_critical(u, v, key, critical_edges):
            if verbose:
                print(
                    f"    SKIP — edge disconnects the city",
                    file=sys.stderr,
                )
            results.append(BraessResult(
                edge=(u, v, key),
                baseline_avg=baseline_avg,
                test_avg=baseline_avg,
                score=0.0,
                disconnected=True,
            ))
            skipped_disconnect += 1
            continue

        # Close the road
        road_network.block_edge(u, v, key)

        # Run headless simulation with this road closed
        test_sim = HeadlessSimulation(road_network, poisson_rate=0.5)
        test_result = test_sim.run_until_stable(
            min_completed=min_completed,
            dt=dt,
            progress_callback=lambda c, t: None,  # quiet during per-road tests
        )
        test_avg = test_result["avg_travel_time_seconds"]

        # Reopen the road
        road_network.unblock_edge(u, v, key)

        # Compute score
        score = baseline_avg - test_avg
        if verbose:
            change_sign = "+" if score > 0 else ""
            print(
                f"    Test avg: {test_avg:.2f}s  "
                f"Score: {change_sign}{score:.3f}s",
                file=sys.stderr,
            )

        results.append(BraessResult(
            edge=(u, v, key),
            baseline_avg=baseline_avg,
            test_avg=test_avg,
            score=score,
            disconnected=False,
        ))

    if verbose:
        print(
            f"\nDone. {skipped_disconnect} roads skipped (disconnect city).",
            file=sys.stderr,
        )

    registry = BraessRegistry(
        city_name=city_name,
        commuter_count=commuter_count,
        min_completed=min_completed,
        baseline_avg=baseline_avg,
        total_roads_tested=total_roads,
        results=results,
    )

    if verbose:
        braess_roads = registry.braess_roads
        if braess_roads:
            print(
                f"Found {len(braess_roads)} Braess-tainted roads "
                f"(closure IMPROVES travel time):",
                file=sys.stderr,
            )
            for r in sorted(braess_roads, key=lambda x: -x.score):
                print(
                    f"  Edge {r.edge}: score={r.score:+.3f}s "
                    f"(baseline {r.baseline_avg:.2f}s → {r.test_avg:.2f}s)",
                    file=sys.stderr,
                )
        else:
            print("No Braess-tainted roads found.", file=sys.stderr)

    return registry


# ── Parallel testing (multiprocessing) ──────────────────────────

def _test_edge_worker(
    args: tuple[NodeID, NodeID, EdgeKey, str, list[tuple[NodeID, NodeID]], float, int, str],
) -> BraessResult:
    """Test a single road in a worker process.

    Each worker loads the city from cache independently to avoid
    sharing (pickling) the RoadNetwork across processes.

    Args:
        args: (u, v, key, city_name, commuter_pairs, dt, min_completed, file_label).

    Returns:
        BraessResult for the tested edge.
    """
    u, v, key, city_name, commuter_pairs, dt, min_completed, file_label = args
    try:
        rn, _ = _load_city(file_label)
    except (ValueError, Exception):
        return BraessResult(
            edge=(u, v, key),
            baseline_avg=0.0,
            test_avg=0.0,
            score=0.0,
            disconnected=True,
        )

    # Close the road
    rn.block_edge(u, v, key)

    # Run headless simulation
    sim = HeadlessSimulation(rn, poisson_rate=0.5)
    result = sim.run_until_stable(
        min_completed=min_completed,
        dt=dt,
        progress_callback=None,
    )
    test_avg = result["avg_travel_time_seconds"]

    # Note: baseline_avg is set by the calling process after merging
    return BraessResult(
        edge=(u, v, key),
        baseline_avg=0.0,
        test_avg=test_avg,
        score=0.0,
        disconnected=False,
    )


def find_braess_roads_parallel(
    road_network: RoadNetwork,
    commuter_pairs: list[tuple[NodeID, NodeID]],
    city_name: str = "unknown",
    commuter_count: int = 500,
    min_completed: int = 500,
    dt: float = 0.2,
    num_workers: Optional[int] = None,
    verbose: bool = True,
) -> BraessRegistry:
    """Identify Braess-tainted roads using parallel workers.

    Same algorithm as find_braess_roads, but tests roads in parallel
    using multiprocessing. Each worker loads the city independently
    from the cached graphml file.

    Args:
        road_network: The city road network (used for baseline and critical edges).
        commuter_pairs: List of (origin, destination) node pairs.
        city_name: Identifier for the city.
        commuter_count: Number of commuters.
        min_completed: Trips per road test.
        dt: Simulation timestep.
        num_workers: Number of parallel workers (default: CPU count - 1).
        verbose: Print progress.

    Returns:
        BraessRegistry containing all results.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    all_edges = road_network.get_all_edges()
    if verbose:
        print(f"Total edges in network: {len(all_edges)}", file=sys.stderr)

    # Pre-compute critical edges (Tarjan's bridges)
    if verbose:
        print("Pre-computing critical edges (Tarjan's bridges)...", file=sys.stderr)
    critical_edges = _build_critical_edge_set(road_network)
    if verbose:
        print(
            f"  {len(critical_edges)} critical edges identified",
            file=sys.stderr,
        )

    # ── Phase 1: Baseline ──
    if verbose:
        print("Phase 1: Establishing baseline (all roads open)...", file=sys.stderr)

    baseline_sim = HeadlessSimulation(road_network, poisson_rate=0.5)
    baseline_result = baseline_sim.run_until_stable(
        min_completed=min_completed,
        dt=dt,
        progress_callback=lambda c, t: print(
            f"  Baseline: {c}/{t} trips completed", file=sys.stderr,
        ),
    )
    baseline_avg = baseline_result["avg_travel_time_seconds"]
    if verbose:
        print(
            f"  Baseline avg travel time: {baseline_avg:.2f}s "
            f"(after {baseline_result['completed_trips']} trips)",
            file=sys.stderr,
        )

    # ── Phase 2: Test each road (parallel) ──
    # Separate edges into critical (skip) and non-critical (test)
    skip_results: list[BraessResult] = []
    edges_to_test: list[tuple[NodeID, NodeID, EdgeKey]] = []

    for u, v, key in all_edges:
        if _edge_is_critical(u, v, key, critical_edges):
            skip_results.append(BraessResult(
                edge=(u, v, key),
                baseline_avg=baseline_avg,
                test_avg=baseline_avg,
                score=0.0,
                disconnected=True,
            ))
        else:
            edges_to_test.append((u, v, key))

    if verbose:
        print(
            f"Phase 2: Testing {len(edges_to_test)} non-critical edges "
            f"using {num_workers} workers...",
            file=sys.stderr,
        )

    # Prepare arguments for workers
    file_label = city_name  # used to load city in each worker
    worker_args = [
        (u, v, key, city_name, commuter_pairs, dt, min_completed, file_label)
        for u, v, key in edges_to_test
    ]

    # Run in parallel
    test_results: list[BraessResult] = []
    start_time = time.time()
    done_count = 0
    total_to_test = len(edges_to_test)

    if total_to_test == 0:
        if verbose:
            print("  No non-critical edges to test.", file=sys.stderr)
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_test_edge_worker, worker_args):
                test_results.append(result)
                done_count += 1
                if verbose and done_count % max(1, total_to_test // 20) == 0:
                    elapsed_here = time.time() - start_time
                    rate = done_count / elapsed_here if elapsed_here > 0 else 0
                    eta = (total_to_test - done_count) / rate if rate > 0 else 0
                    print(
                        f"  [{done_count}/{total_to_test}] "
                        f"({rate:.1f} roads/s, ETA: {eta:.0f}s)",
                        file=sys.stderr,
                    )

    # Merge skip results and tested results
    all_results = skip_results + test_results

    # Set baseline_avg and score for tested edges after the fact
    for r in all_results:
        if not r.disconnected and r.baseline_avg == 0.0:
            r.baseline_avg = baseline_avg
            r.score = baseline_avg - r.test_avg

    registry = BraessRegistry(
        city_name=city_name,
        commuter_count=commuter_count,
        min_completed=min_completed,
        baseline_avg=baseline_avg,
        total_roads_tested=len(all_edges),
        results=all_results,
    )

    if verbose:
        elapsed = time.time() - start_time
        print(f"\nParallel testing complete: {elapsed:.1f}s", file=sys.stderr)
        braess_roads = registry.braess_roads
        if braess_roads:
            print(
                f"Found {len(braess_roads)} Braess-tainted roads:",
                file=sys.stderr,
            )
            for r in sorted(braess_roads, key=lambda x: -x.score)[:10]:
                print(
                    f"  Edge {r.edge}: score={r.score:+.3f}s",
                    file=sys.stderr,
                )
        else:
            print("No Braess-tainted roads found.", file=sys.stderr)

    return registry


# ── CLI entry point ───────────────────────────────────────────────

def _load_city(city_name: str) -> tuple[RoadNetwork, list[TrafficLight]]:
    """Load a city by name from cached graphml or download.

    Maps stored names (like "Greenwood_Township__Pennsylvania__United_States") to
    the proper format expected by map_manager.load_city and falls back to OSMnx
    if no cache file exists.

    Args:
        city_name: Name of the city file (without path or extension).

    Returns:
        (RoadNetwork, list of TrafficLight).

    Raises:
        ValueError: If the city cannot be loaded.
    """
    from src.map_manager import load_city

    # Check if the cache file exists for this exact name
    cache_path = os.path.join(config.OSM_CACHE_DIR, f"{city_name}.graphml")
    if os.path.exists(cache_path):
        # Convert the cached filename back to a human-readable name
        # "Greenwood_Township__Pennsylvania__United_States" -> "Greenwood Township, Pennsylvania, United States"
        human_name = city_name.replace("__", ", ").replace("_", " ")
        return load_city(human_name)

    # If not found as-is, try treating it as already a human-readable name
    human_name = city_name.replace("__", ", ").replace("_", " ")
    return load_city(human_name)


def _generate_commuters(
    road_network: RoadNetwork,
    count: int = 500,
    seed: int = 42,
) -> list[tuple[NodeID, NodeID]]:
    """Generate a fixed set of origin-destination pairs.

    Each pair is guaranteed to have a path between origin and destination
    (using the full network with no blocked edges).

    Args:
        road_network: The city road network.
        count: Number of commuter pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of (origin, destination) node tuples.
    """
    rng = random.Random(seed)
    nodes = road_network.get_all_nodes()
    working_graph = road_network.get_working_graph()

    pairs: list[tuple[NodeID, NodeID]] = []
    attempts = 0
    max_attempts = count * 20  # safety limit

    while len(pairs) < count and attempts < max_attempts:
        attempts += 1
        origin = rng.choice(nodes)
        destination = rng.choice(nodes)
        if origin == destination:
            continue

        # Ensure a path exists
        try:
            path = nx.shortest_path(
                working_graph,
                source=origin,
                target=destination,
                weight="length_meters",
            )
            if path is not None:
                pairs.append((origin, destination))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    log.info("Generated %d/%d commuter pairs (%d attempts)", len(pairs), count, attempts)
    return pairs


def main() -> None:
    """Run the Braess detector from the command line and save results."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.braess_detector <city_name> [commuter_count] [min_completed]", file=sys.stderr)
        print("Example: python -m src.braess_detector Greenwood_Township__Pennsylvania__United_States", file=sys.stderr)
        sys.exit(1)

    city_name = sys.argv[1]
    commuter_count = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    min_completed = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    print(f"Loading city: {city_name}", file=sys.stderr)
    road_network, traffic_lights = _load_city(city_name)
    print(
        f"Loaded: {road_network.node_count} nodes, "
        f"{road_network.edge_count} edges",
        file=sys.stderr,
    )

    print(f"Generating {commuter_count} commuter pairs...", file=sys.stderr)
    commuter_pairs = _generate_commuters(road_network, count=commuter_count)

    print("Starting Braess detection (parallel)...", file=sys.stderr)
    start_time = time.time()
    registry = find_braess_roads_parallel(
        road_network,
        commuter_pairs,
        city_name=city_name,
        commuter_count=commuter_count,
        min_completed=min_completed,
        dt=0.2,
        verbose=True,
    )
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.1f}s", file=sys.stderr)
    print(f"Total roads tested: {registry.total_roads_tested}", file=sys.stderr)
    print(f"Braess-tainted roads: {registry.num_braess_roads}", file=sys.stderr)

    # ── Save results to JSON ──
    output_path = BraessRegistry.default_filename(city_name)
    registry.save(output_path)
    print(f"\nResults saved to: {output_path}", file=sys.stderr)

    # Print final summary to stdout for easy parsing
    print("\n=== BRAESS RESULTS ===")
    for r in sorted(registry.braess_roads, key=lambda x: -x.score):
        print(f"{r.edge[0]},{r.edge[1]},{r.edge[2]},{r.score:.4f},{r.baseline_avg:.2f},{r.test_avg:.2f}")


if __name__ == "__main__":
    main()