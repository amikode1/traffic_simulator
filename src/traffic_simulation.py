"""Core simulation loop — spawns, moves, and manages cars and traffic lights.

This is the central orchestrator that ties together the road network,
pathfinding, cars, and traffic lights. It runs the simulation tick and
provides methods for the renderer and UI to query state.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

import config
from src.car import Car
from src.pathfinding import find_path, compute_path_length, build_bpr_graph, find_on_bpr_graph
from src.road_network import NodeID, EdgeKey, RoadNetwork
from src.traffic_light import TrafficLight, SignalPhase

log = logging.getLogger(__name__)


@dataclass
class TrafficSimulation:
    """The main simulation engine.

    Attributes:
        road_network: The RoadNetwork instance.
        traffic_lights: List of TrafficLight instances.
        cars: List of active Car instances.
        algorithm: Currently selected pathfinding algorithm.
        desired_car_count: Target number of cars (set by UI slider).
        next_car_id: Auto-incrementing ID counter.
        time_seconds: Total elapsed simulation time.
        spawn_timer: Accumulator for spawning cars at intervals.
    """

    road_network: RoadNetwork
    traffic_lights: list[TrafficLight] = field(default_factory=list)
    cars: list[Car] = field(default_factory=list)
    algorithm: str = config.DEFAULT_ALGORITHM
    desired_car_count: int = config.DEFAULT_CAR_COUNT
    speed_multiplier: float = config.DEFAULT_SPEED_MULTIPLIER
    next_car_id: int = 0
    time_seconds: float = 0.0
    spawn_timer: float = 0.0
    _stats: dict = field(default_factory=lambda: {
        "total_spawned": 0,
        "total_reached_destination": 0,
        "total_rerouted": 0,
        "total_travel_time_seconds": 0.0,
        "completed_trips": 0,
    })
    _selfish_timer: float = 0.0
    _selfish_reroute_queue: list[int] = field(default_factory=list)
    _selfish_reroute_idx: int = 0
    _selfish_counts_snapshot: dict = field(default_factory=dict)
    _selfish_bpr_graph: Any = None  # pre-built nx.DiGraph for selfish cycle

    # ── Public API ──────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """Advance the simulation by *dt* seconds.

        Args:
            dt: Delta time in seconds (should be capped to avoid spiral of death).
        """
        dt = min(dt, 0.1)  # cap at 100ms to prevent physics explosion
        self.time_seconds += dt

        # 0. Safety sweep — stop any car that snuck onto a blocked edge
        self._sweep_blocked_edges()

        # 0b. Selfish algorithm: periodic reroute based on congestion
        if self.algorithm == "selfish":
            self._selfish_timer += dt
            if self._selfish_timer >= config.SELFISH_REROUTE_INTERVAL:
                self._selfish_timer = 0.0
                self._reroute_all_cars_selfish()
            # Process a small batch every tick (never freeze)
            self._process_selfish_queue()

        # 1. Update traffic lights
        self._update_traffic_lights(dt)

        # 2. Spawn or despawn cars to match desired count
        self._adjust_car_count(dt)

        # 3. Move all cars
        self._move_cars(dt)

        # 4. Remove cars that reached destination
        self._remove_arrived_cars()

    def set_desired_car_count(self, count: int) -> None:
        """Set the target number of cars (clamped to min/max)."""
        self.desired_car_count = max(
            config.MIN_CAR_COUNT, min(count, config.MAX_CAR_COUNT)
        )

    def set_speed_multiplier(self, multiplier: float) -> None:
        """Set the simulation speed multiplier (clamped to min/max)."""
        self.speed_multiplier = max(
            config.MIN_SPEED_MULTIPLIER,
            min(multiplier, config.MAX_SPEED_MULTIPLIER),
        )

    def set_algorithm(self, algorithm: str) -> None:
        """Change the pathfinding algorithm and reroute all cars."""
        if algorithm in config.AVAILABLE_ALGORITHMS:
            self.algorithm = algorithm
            self._reroute_all_cars()

    def toggle_block_edge(self, u: NodeID, v: NodeID, key: EdgeKey = 0) -> bool:
        """Toggle a road edge's blocked status. Returns new blocked state.

        If the road is two-way (both (u,v) and (v,u) exist), toggling one
        direction toggles both — the whole road is blocked or open.
        """
        is_blocked = self.road_network.toggle_block_edge(u, v, key)
        # Also toggle the opposite direction if it's a two-way road
        if self.road_network.get_edge(v, u, key) is not None:
            if is_blocked:
                self.road_network.block_edge(v, u, key)
            else:
                self.road_network.unblock_edge(v, u, key)
        # Reroute cars affected by this blockage (both directions)
        self._reroute_affected_cars(u, v, key)
        if self.road_network.get_edge(v, u, key) is not None:
            self._reroute_affected_cars(v, u, key)
        return is_blocked

    def block_edge(self, u: NodeID, v: NodeID, key: EdgeKey = 0) -> bool:
        """Block a road edge. Returns True if newly blocked.

        If the road is two-way (both (u,v) and (v,u) exist), blocking one
        direction blocks both — the whole road is closed.
        """
        result = self.road_network.block_edge(u, v, key)
        if result:
            # Also block the opposite direction if two-way
            if self.road_network.get_edge(v, u, key) is not None:
                self.road_network.block_edge(v, u, key)
            self._reroute_affected_cars(u, v, key)
            if self.road_network.get_edge(v, u, key) is not None:
                self._reroute_affected_cars(v, u, key)
        return result

    def unblock_edge(self, u: NodeID, v: NodeID, key: EdgeKey = 0) -> bool:
        """Unblock a road edge. Returns True if it was blocked.

        If the road is two-way, unblocking one direction unblocks both."""
        result = self.road_network.unblock_edge(u, v, key)
        if result:
            if self.road_network.get_edge(v, u, key) is not None:
                self.road_network.unblock_edge(v, u, key)
        return result

    def get_stats(self) -> dict:
        """Return a copy of the simulation statistics, including computed averages."""
        stats = dict(self._stats)
        if stats["completed_trips"] > 0:
            stats["avg_travel_time_seconds"] = (
                stats["total_travel_time_seconds"] / stats["completed_trips"]
            )
        else:
            stats["avg_travel_time_seconds"] = 0.0
        return stats

    def _get_car_counts(self) -> dict[tuple[NodeID, NodeID, EdgeKey], int]:
        """Count cars currently on each edge.

        Returns a dict mapping (u, v, key) -> number of cars on that edge.
        Used by the selfish algorithm to compute BPR congestion costs.
        """
        counts: dict[tuple[NodeID, NodeID, EdgeKey], int] = {}
        for car in self.cars:
            edge = car.current_edge
            counts[edge] = counts.get(edge, 0) + 1
        return counts

    def _reroute_all_cars_selfish(self) -> None:
        """Start a staggered selfish reroute cycle.

        Builds the BPR-weighted graph ONCE (the expensive part), then queues
        all car IDs for pathfinding. The actual Dijkstra calls are spread
        across many frames.
        """
        working_graph = self.road_network.get_working_graph()
        self._selfish_counts_snapshot = self._get_car_counts()
        self._selfish_bpr_graph = build_bpr_graph(
            working_graph, self._selfish_counts_snapshot,
        )
        self._selfish_reroute_queue = [c.id for c in self.cars]
        self._selfish_reroute_idx = 0
        log.info(
            "Selfish reroute queued for %d cars (BPR graph: %d edges)",
            len(self._selfish_reroute_queue),
            self._selfish_bpr_graph.number_of_edges(),
        )

    def _process_selfish_queue(self) -> None:
        """Process a batch of pending selfish reroutes (non-blocking).

        Called every tick. Processes BATCH_SIZE cars from the queue
        using the pre-built BPR graph so no conversion is done per car.
        """
        BATCH_SIZE = 60  # cars per tick – Dijkstra on DiGraph is fast
        queue = self._selfish_reroute_queue
        if not queue or self._selfish_reroute_idx >= len(queue):
            return

        bpr_graph = self._selfish_bpr_graph
        if bpr_graph is None:
            return  # shouldn't happen

        end = min(self._selfish_reroute_idx + BATCH_SIZE, len(queue))
        batch_ids = queue[self._selfish_reroute_idx:end]
        self._selfish_reroute_idx = end

        for car_id in batch_ids:
            car = next((c for c in self.cars if c.id == car_id), None)
            if car is not None:
                self._reroute_car(car, bpr_graph=bpr_graph)

        if self._selfish_reroute_idx >= len(queue):
            log.debug("Selfish reroute cycle completed (all %d cars)", len(queue))

    # ── Internal: Spawning ──────────────────────────────────────

    def _spawn_car(self) -> Optional[Car]:
        """Spawn a single car at a random edge on the network.

        The car's route always starts with its current edge (u → v),
        ensuring movement is consistent from frame one.

        Returns:
            The new Car, or None if spawning failed.
        """
        edges = self.road_network.get_all_edges()
        if not edges:
            return None

        working_graph = self.road_network.get_working_graph()
        nodes = self.road_network.get_all_nodes()
        if len(nodes) < 2:
            return None

        # Try random edges until we find one with a valid path
        shuffled_edges = random.sample(edges, min(len(edges), 50))
        for edge in shuffled_edges:
            u, v, key = edge

            # Never spawn on a blocked road
            if self.road_network.is_edge_blocked(u, v, key):
                continue

            # Pick a random destination (not u)
            destination = random.choice(nodes)
            attempts = 0
            while destination == u and attempts < 30:
                destination = random.choice(nodes)
                attempts += 1
            if destination == u:
                continue

            # Find a path from u to destination
            path = find_path(working_graph, u, destination, self.algorithm)
            if path is None:
                continue

            # Flatten path list
            if isinstance(path, list) and path and isinstance(path[0], list):
                path = path[0]

            # Ensure route starts with [u, v]
            if len(path) < 2:
                continue
            if path[0] != u:
                continue
            if path[1] == v:
                # Perfect — the path already uses our edge
                break
            # Path doesn't go through v from u. Try prepending [u, v].
            # Find a path from v to the destination
            path_from_v = find_path(working_graph, v, path[-1], self.algorithm)
            if path_from_v is None:
                continue
            if isinstance(path_from_v, list) and path_from_v and isinstance(path_from_v[0], list):
                path_from_v = path_from_v[0]
            if len(path_from_v) < 2:
                continue
            # Build route: u → v → rest of path_from_v.
            # Final safety-check: (u, v) must still be unblocked.
            if self.road_network.is_edge_blocked(u, v, key):
                continue
            path = [u, v] + path_from_v[1:]
            break
        else:
            return None  # No valid edge found

        # Determine lane and speed
        edge_data = self.road_network.get_edge_attributes(u, v, key)
        lane = random.randint(0, max(edge_data.lanes - 1, 0))
        target_speed = edge_data.speed_kph / 3.6

        car = Car(
            id=self.next_car_id,
            current_edge=(u, v, key),
            current_lane=lane,
            progress=random.random() * 0.3,  # scatter near the start
            target_speed_ms=target_speed,
            speed_ms=target_speed * (0.7 + 0.3 * random.random()),
            color=config.CAR_COLOR,
            spawn_time=self.time_seconds,
        )
        car.route = path
        car.route_index = 0  # route[0] = u, route[1] = v

        self.next_car_id += 1
        self._stats["total_spawned"] += 1
        return car

    def _adjust_car_count(self, dt: float) -> None:
        """Spawn or despawn cars to match the desired count."""
        # Spawn — burst spawn when far from target
        if len(self.cars) < self.desired_car_count:
            deficit = self.desired_car_count - len(self.cars)
            # Spawn multiple cars per tick when deficit is large
            spawn_count = 1
            if deficit >= 20:
                spawn_count = 3
            elif deficit >= 10:
                spawn_count = 2
            for _ in range(spawn_count):
                if len(self.cars) < self.desired_car_count:
                    car = self._spawn_car()
                    if car is not None:
                        self.cars.append(car)
                    else:
                        break  # can't spawn right now

        # Despawn (remove excess cars)
        while len(self.cars) > self.desired_car_count:
            if self.cars:
                self.cars.pop(0)  # remove oldest car
            else:
                break

    # ── Internal: Movement ──────────────────────────────────────

    def _update_traffic_lights(self, dt: float) -> None:
        """Update all traffic light phases."""
        for light in self.traffic_lights:
            light.update(dt)

    def _move_cars(self, dt: float) -> None:
        """Move all cars along their routes."""
        for car in self.cars[:]:  # iterate over copy
            self._move_single_car(car, dt)

    # ── Car-following helpers ────────────────────────────────────

    def _get_lead_car(self, car: Car) -> Optional[Car]:
        """Find the nearest car directly ahead on the same edge and lane.

        Returns:
            The Car immediately ahead, or None if this car is at the front
            of its lane.
        """
        target_edge = car.current_edge
        target_lane = car.current_lane

        lead: Optional[Car] = None
        for other in self.cars:
            if other.id == car.id:
                continue
            if other.current_edge == target_edge and other.current_lane == target_lane:
                if other.progress > car.progress:
                    if lead is None or other.progress < lead.progress:
                        lead = other
        return lead

    def _apply_car_following(
        self, car: Car, lead_car: Car, edge_length: float, dt: float
    ) -> bool:
        """Adjust *car*'s speed based on the lead car ahead on the same lane.

        Three zones based on gap relative to desired_gap:
          Zone 1 — gap < MIN_GAP:                 emergency brake to 0
          Zone 2 — MIN_GAP ≤ gap < desired_gap:   decelerate toward leader
          Zone 3 — desired_gap ≤ gap < LOOK_AHEAD*desired_gap: gentle match
          Zone 4 — gap ≥ LOOK_AHEAD*desired_gap: no influence

        Args:
            car: The following car (behind).
            lead_car: The car ahead (in front).
            edge_length: Length of the current edge in metres.
            dt: Delta time in seconds.

        Returns:
            True if the car's speed was constrained (zones 1-3).
            False if the gap was so large no adjustment was needed (zone 4).
        """
        # Bumper-to-bumper gap (metres)
        gap_meters = (
            (lead_car.progress - car.progress) * edge_length
            - config.CAR_LENGTH_METERS
        )
        lead_speed = lead_car.speed_ms

        desired_gap = config.MIN_GAP_METERS + config.FOLLOW_TIME_HEADWAY * car.speed_ms

        # ── Zone 1: overlapping / below minimum gap ──
        if gap_meters <= config.MIN_GAP_METERS:
            car.speed_ms = max(car.speed_ms - config.EMERGENCY_DECELERATION * dt, 0.0)
            if gap_meters <= 0:
                car.speed_ms = 0.0
            car.waiting_for_car_ahead = car.speed_ms < 0.5
            if car.waiting_for_car_ahead:
                car.car_ahead_wait_seconds += dt
            return True

        # ── Zone 2: braking zone ──
        if gap_meters < desired_gap:
            gap_ratio = gap_meters / max(desired_gap, 0.001)
            target_speed = lead_speed * (gap_ratio ** 0.5)

            if car.speed_ms > target_speed:
                car.speed_ms = max(
                    car.speed_ms - config.COMFORTABLE_DECELERATION * dt,
                    target_speed, 0.0,
                )

            car.waiting_for_car_ahead = car.speed_ms < car.target_speed_ms * 0.3
            if car.waiting_for_car_ahead:
                car.car_ahead_wait_seconds += dt
            else:
                car.car_ahead_wait_seconds = 0.0
            return True

        # ── Zone 3: moderate gap — match speed gently ──
        if gap_meters < desired_gap * config.LOOK_AHEAD_FACTOR:
            if car.speed_ms > lead_speed + config.MAX_CATCHUP_SPEED_DELTA:
                car.speed_ms = max(
                    car.speed_ms - config.COMFORTABLE_DECELERATION * dt,
                    lead_speed,
                )
            car.waiting_for_car_ahead = False
            car.car_ahead_wait_seconds = 0.0
            return True

        # ── Zone 4: gap is large — no influence ──
        return False

    # ── Movement ─────────────────────────────────────────────────

    def _move_single_car(self, car: Car, dt: float) -> None:
        """Move a single car along its route.

        Handles:
          1. Traffic light stopping — stop for red lights at intersections.
          2. Blocked-edge detection — reroute if next road is blocked.
          3. Car-following — queue behind the car ahead on the same lane,
             overriding normal acceleration when a lead car is nearby.
          4. Normal acceleration + advance (only when no lead car constrains).
          5. Edge transition — move to next edge at progress ≥ 1.0,
             checking that there is room on the target edge.
        """
        if car.reached_destination:
            return

        u, v, key = car.current_edge
        edge_data = self.road_network.get_edge_attributes(u, v, key)
        edge_length = edge_data.length_meters

        if edge_length <= 0:
            return

        # ── 0. If current edge is blocked, do NOT advance ────────
        if self.road_network.is_edge_blocked(u, v, key):
            # Reroute already happened via _reroute_affected_cars,
            # but the car may still be on this edge (e.g. stranded).
            # Stop the car — it cannot use a closed road.
            car.speed_ms = 0.0
            return

        # ── 1. Traffic light at the target node ──────────────────
        if self._should_stop_at_light(car, v):
            car.waiting_at_light = True
            car.light_wait_seconds += dt
            car.speed_ms = 0.0
            car.waiting_for_car_ahead = False
            car.car_ahead_wait_seconds = 0.0
            return
        car.waiting_at_light = False
        car.light_wait_seconds = 0.0

        # ── 2. Check if the next edge is blocked or unreachable ──
        next_edge = self._get_next_edge(car)
        if next_edge is not None:
            nu, nv, nkey = next_edge
            if self.road_network.is_edge_blocked(nu, nv, nkey):
                self._reroute_car(car)
                return
        elif car.route_index + 2 < len(car.route):
            # All parallel edges to the next route node are blocked.
            # Try a full reroute to find a different path.
            self._reroute_car(car)
            return

        # ── 3. Car-following (same edge, same lane) ──────────────
        lead_car = self._get_lead_car(car)

        if lead_car is not None:
            constrained = self._apply_car_following(car, lead_car, edge_length, dt)
            if constrained:
                # ── Check next edge blocked before advancing ──
                next_edge = self._get_next_edge(car)
                if next_edge is not None:
                    nu, nv, nkey = next_edge
                    if self.road_network.is_edge_blocked(nu, nv, nkey):
                        self._reroute_car(car)
                        return
                elif car.route_index + 2 < len(car.route):
                    # All parallel edges blocked — try full reroute
                    self._reroute_car(car)
                    return

                # Car-following adjusted speed — advance and check transition
                distance = car.speed_ms * dt
                car.advance_progress(distance, edge_length)
                self._try_edge_transition(car, dt)
                return
            else:
                # Lead car exists but gap is large (zone 4) — fall through
                # to normal acceleration below; reset waiting state.
                car.waiting_for_car_ahead = False
                car.car_ahead_wait_seconds = 0.0

        # ── 4. No constraining lead car — accelerate normally ────
        car.waiting_for_car_ahead = False
        car.car_ahead_wait_seconds = 0.0
        car.update(dt)

        # ── 5. Advance the car along the edge ────────────────────
        distance = car.speed_ms * dt
        car.advance_progress(distance, edge_length)

        # ── 6. Edge transition (progress ≥ 1.0) ──────────────────
        self._try_edge_transition(car, dt)

    def _try_edge_transition(self, car: Car, dt: float) -> None:
        """Attempt to transition the car to the next edge when progress ≥ 1.0.

        Refuses to switch onto a blocked road — the car waits at the
        intersection until the road is unblocked or an alternative appears.

        Also checks that there is room on the target edge (same lane, near
        start) before allowing the switch.

        Args:
            car: The car that may be ready to transition.
            dt: Delta time in seconds.
        """
        if car.progress < 1.0:
            return

        car.progress = 1.0

        next_edge = self._get_next_edge(car)
        if next_edge is None:
            if car.route_index + 2 < len(car.route):
                # All parallel edges to the next route node are blocked.
                # Try a full reroute to find a different path.
                self._reroute_car(car)
            return  # reached destination or waiting for reroute

        nu, nv, nkey = next_edge

        # ── Never drive onto a blocked road ──
        if self.road_network.is_edge_blocked(nu, nv, nkey):
            car.speed_ms = 0.0
            car.waiting_for_car_ahead = True
            car.car_ahead_wait_seconds += dt
            return

        next_edge_data = self.road_network.get_edge_attributes(nu, nv, nkey)
        next_edge_length = next_edge_data.length_meters
        lane = random.randint(0, max(next_edge_data.lanes - 1, 0))

        # Check if there is room on the target edge (near the start)
        if next_edge_length > 0:
            for other in self.cars:
                if other.id == car.id:
                    continue
                if other.current_edge == (nu, nv, nkey) and other.current_lane == lane:
                    other_dist = other.progress * next_edge_length
                    needed_room = config.CAR_LENGTH_METERS + config.MIN_GAP_METERS
                    if other_dist < needed_room:
                        # Not enough room — wait at current position
                        car.speed_ms = 0.0
                        car.waiting_for_car_ahead = True
                        car.car_ahead_wait_seconds += dt
                        return

        car.switch_to_next_edge(next_edge, lane)

    def _should_stop_at_light(self, car: Car, target_node: NodeID) -> bool:
        """Check if the car should stop at a red light ahead.

        Only stops if the car is close enough to the intersection.
        """
        for light in self.traffic_lights:
            if light.node_id == target_node and light.is_red():
                # Only stop if close to the intersection
                if car.progress > 0.85:
                    # Additional check: is this light relevant for this edge?
                    if not light.affected_edges or car.current_edge in light.affected_edges:
                        return True
        return False

    def _get_next_edge(self, car: Car) -> Optional[tuple[NodeID, NodeID, EdgeKey]]:
        """Get the next unblocked edge the car should drive on, based on its route.

        The car is currently on edge (route[i], route[i+1]) heading toward route[i+1].
        The next edge is (route[i+1], route[i+2]).

        Skips blocked edges — if the only connection is blocked, returns None
        so the caller will keep the car waiting.
        """
        if car.route_index + 2 >= len(car.route):
            return None

        current_node = car.route[car.route_index + 1]  # node we're arriving at
        next_node = car.route[car.route_index + 2]      # node after that

        # Find the edge from current_node to next_node
        edges = self.road_network.edges_from_node(current_node)
        for u, v, key in edges:
            if v == next_node and not self.road_network.is_edge_blocked(u, v, key):
                return (u, v, key)

        # All matching edges are blocked — caller will handle waiting
        return None

    # ── Internal: Rerouting ─────────────────────────────────────

    def _reroute_car(self, car: Car,
                     car_counts: Optional[dict] = None,
                     bpr_graph: Any = None) -> None:
        """Recalculate a car's route when it encounters a blocked road.

        If a pre-built BPR graph is provided (the fast path from a selfish
        reroute cycle), it is used directly — no conversion overhead.

        If the car's *current* edge no longer aligns with the new route,
        the car is teleported onto the correct first edge of the new path.

        Args:
            car: The car to reroute.
            car_counts: Optional pre-computed edge->car dict for selfish algo.
            bpr_graph: Pre-built BPR-weighted DiGraph (from build_bpr_graph).
                When provided, find_on_bpr_graph is used — the fast path.
        """
        working_graph = self.road_network.get_working_graph()
        u, v, key = car.current_edge
        is_current_blocked = self.road_network.is_edge_blocked(u, v, key)

        if len(car.route) < 2:
            return
        destination = car.route[-1]

        # ── Find alternative path ──────────────────────────────
        # Fast path: pre-built BPR graph (bulk selfish reroute)
        if bpr_graph is not None:
            if is_current_blocked:
                path = find_on_bpr_graph(bpr_graph, u, destination)
            else:
                path = find_on_bpr_graph(bpr_graph, v, destination)
                if path is None:
                    path = find_on_bpr_graph(bpr_graph, u, destination)
        # Slow path: build on demand (single-car reroute)
        else:
            kwargs = {}
            if self.algorithm == "selfish":
                if car_counts is None:
                    car_counts = self._get_car_counts()
                kwargs["car_counts"] = car_counts

            if is_current_blocked:
                path = find_path(working_graph, u, destination,
                                 self.algorithm, **kwargs)
            else:
                path = find_path(working_graph, v, destination,
                                 self.algorithm, **kwargs)
                if path is None:
                    path = find_path(working_graph, u, destination,
                                     self.algorithm, **kwargs)

        if path is None:
            log.debug("Car %d stranded — no alternative route found", car.id)
            car.color = config.CAR_HIGHLIGHT_COLOR
            return

        if isinstance(path, list) and path and isinstance(path[0], list):
            path = path[0]

        # ── Build new route starting from u ──
        if path[0] == u:
            new_route = path
        elif path[0] == v:
            new_route = [u] + path
        else:
            new_route = [u, v] + path

        car.route = new_route
        car.route_index = 0
        car.rerouting = True
        car.color = config.CAR_HIGHLIGHT_COLOR
        self._stats["total_rerouted"] += 1

        # ── Teleport if current edge doesn't match new route's first step ──
        needs_teleport = is_current_blocked or (
            len(car.route) >= 2 and car.route[1] != v
        )

        if needs_teleport and len(car.route) >= 2:
            route_u, route_v = car.route[0], car.route[1]
            edges = self.road_network.edges_from_node(route_u)
            new_edge = None
            for eu, ev, ekey in edges:
                if ev == route_v and not self.road_network.is_edge_blocked(eu, ev, ekey):
                    new_edge = (eu, ev, ekey)
                    break
            if new_edge is not None:
                car.current_edge = new_edge
                car.progress = 0.0
                log.debug("Car %d teleported onto %s", car.id, new_edge)

    def _reroute_all_cars(self) -> None:
        """Reroute every car in the simulation (e.g., after algorithm change)."""
        log.info("Rerouting all %d cars with algorithm '%s'", len(self.cars), self.algorithm)
        car_counts = self._get_car_counts() if self.algorithm == "selfish" else None
        for car in self.cars:
            self._reroute_car(car, car_counts)

    def _reroute_affected_cars(
        self, u: NodeID, v: NodeID, key: EdgeKey
    ) -> None:
        """Reroute cars that are currently on or heading toward a blocked edge."""
        for car in self.cars:
            # Check if car's current edge is the blocked one (match full tuple)
            if car.current_edge == (u, v, key):
                self._reroute_car(car)
                continue
            # Also check by (u,v) node-pair — catches parallel lanes with
            # different edge keys (e.g. car on (1,2,1) when (1,2,0) is blocked).
            if (car.current_edge[0], car.current_edge[1]) == (u, v):
                self._reroute_car(car)
                continue
            # Check if the car's remaining route includes this edge
            for i in range(car.route_index, len(car.route) - 1):
                if car.route[i] == u and car.route[i + 1] == v:
                    self._reroute_car(car)
                    break

    def _sweep_blocked_edges(self) -> None:
        """Safety sweep — called every tick.

        Finds every car whose current edge or remaining route contains a
        blocked road, stops it, and forces a reroute (which turns the car
        red and teleports it off the blocked road if possible).
        """
        reroute_set: set[int] = set()
        for car in self.cars:
            u, v, key = car.current_edge
            if self.road_network.is_edge_blocked(u, v, key):
                car.speed_ms = 0.0
                reroute_set.add(car.id)
                continue
            # Check if remaining route contains ANY blocked edge
            for i in range(car.route_index, len(car.route) - 1):
                ru, rv = car.route[i], car.route[i + 1]
                # Check all edge keys for this (u,v) pair
                for eu, ev, ekey in self.road_network.edges_from_node(ru):
                    if ev == rv and self.road_network.is_edge_blocked(eu, ev, ekey):
                        reroute_set.add(car.id)
                        break
                if car.id in reroute_set:
                    break

        for car in self.cars:
            if car.id in reroute_set:
                self._reroute_car(car)

    # ── Internal: Cleanup ───────────────────────────────────────

    def _remove_arrived_cars(self) -> None:
        """Remove cars that have reached their destination and record travel times."""
        remaining: list[Car] = []
        for car in self.cars:
            if car.reached_destination:
                self._stats["total_reached_destination"] += 1
                travel_time = self.time_seconds - car.spawn_time
                self._stats["total_travel_time_seconds"] += travel_time
                self._stats["completed_trips"] += 1
            else:
                remaining.append(car)
        self.cars = remaining

    # ── Coordinate conversion ───────────────────────────────────

    def _lon_lat_to_screen(
        self, lon: float, lat: float,
        bounds: tuple[float, float, float, float],
        screen_size: tuple[int, int],
        padding: int = 50,
    ) -> tuple[float, float]:
        """Convert (lon, lat) to screen pixel coordinates.

        Args:
            lon, lat: Geographic coordinates.
            bounds: (min_lon, min_lat, max_lon, max_lat).
            screen_size: (width, height) in pixels.
            padding: Pixels of padding around the map.

        Returns:
            (x, y) pixel coordinates.
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        width, height = screen_size

        # Available drawing area (minus padding)
        draw_w = width - 2 * padding
        draw_h = height - 2 * padding

        # Map aspect ratio
        map_aspect = (max_lon - min_lon) / (max_lat - min_lat) if (max_lat - min_lat) > 0 else 1
        screen_aspect = draw_w / draw_h

        if map_aspect > screen_aspect:
            # Map is wider — fit to width
            scale = draw_w / (max_lon - min_lon) if (max_lon - min_lon) > 0 else 1
            offset_x = padding
            offset_y = padding + (draw_h - (max_lat - min_lat) * scale) / 2
        else:
            # Map is taller — fit to height
            scale = draw_h / (max_lat - min_lat) if (max_lat - min_lat) > 0 else 1
            offset_y = padding
            offset_x = padding + (draw_w - (max_lon - min_lon) * scale) / 2

        x = offset_x + (lon - min_lon) * scale
        y = offset_y + (max_lat - lat) * scale  # invert Y for screen

        return (x, y)

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Return (min_lon, min_lat, max_lon, max_lat) for the network."""
        min_lon = min_lat = float("inf")
        max_lon = max_lat = float("-inf")

        for node_id in self.road_network.get_all_nodes():
            pos = self.road_network.get_node_position(node_id)
            if pos is not None:
                lon, lat = pos
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)

        return (min_lon, min_lat, max_lon, max_lat)