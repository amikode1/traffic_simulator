"""Car entity with lane-level movement, speed limits, and route following.

Each car occupies a specific lane on a road edge and interpolates smoothly
along the road geometry. The car knows its current route (list of nodes)
and recalculates when a blocked road is detected ahead.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from shapely.geometry import LineString, Point

import config
from src.road_network import NodeID, EdgeKey

log = logging.getLogger(__name__)


@dataclass
class Car:
    """A single car in the simulation.

    Attributes:
        id: Unique identifier for this car.
        current_edge: (u, v, key) of the road the car is currently on.
        current_lane: Zero-indexed lane number (0 = rightmost).
        progress: Normalised [0, 1] position along the current edge.
        speed_ms: Current speed in metres per second.
        target_speed_ms: Desired speed (limited by road speed limit).
        route: List of node IDs forming the remaining path.
        route_index: Current position in the route (index of the target node).
        waiting_at_light: Whether the car is stopped at a red light.
        light_wait_seconds: How long the car has been waiting.
        color: Hex colour string for rendering.
        rerouting: Whether the car is currently recalculating its route.
    """

    id: int
    current_edge: tuple[NodeID, NodeID, EdgeKey]
    current_lane: int = 0
    progress: float = 0.0
    speed_ms: float = config.DEFAULT_SPEED_MS
    target_speed_ms: float = config.DEFAULT_SPEED_MS
    route: list[NodeID] = field(default_factory=list)
    route_index: int = 0
    spawn_time: float = 0.0
    waiting_at_light: bool = False
    light_wait_seconds: float = 0.0
    waiting_for_car_ahead: bool = False
    car_ahead_wait_seconds: float = 0.0
    color: str = config.CAR_COLOR
    rerouting: bool = False

    # ── Properties ──────────────────────────────────────────────

    @property
    def is_moving(self) -> bool:
        """Whether the car is currently moving (not waiting at light or for car ahead)."""
        return not (self.waiting_at_light or self.waiting_for_car_ahead) and self.speed_ms > 0.0

    @property
    def reached_destination(self) -> bool:
        """Whether the car has completed its route.

        The last edge is (route[-2], route[-1]). When progress >= 1.0 on
        that edge and there's no next edge, the destination is reached.
        """
        has_no_next_edge = self.route_index + 2 >= len(self.route)
        return has_no_next_edge and self.progress >= 1.0

    @property
    def current_node(self) -> Optional[NodeID]:
        """The source node of the current edge (where the car is coming from)."""
        if not self.route or self.route_index >= len(self.route):
            return None
        return self.route[self.route_index]

    @property
    def target_node(self) -> Optional[NodeID]:
        """The target node of the current edge (where the car is heading)."""
        if self.route_index + 1 < len(self.route):
            return self.route[self.route_index + 1]
        return None

    # ── Movement ────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """Advance the car's speed towards target speed (acceleration).

        Does NOT handle car-following adjustments — that is managed by the
        simulation layer which overrides speed_ms directly when following.
        """
        if self.waiting_at_light or self.waiting_for_car_ahead or self.reached_destination:
            return

        # Accelerate towards target speed
        accel = config.COMFORTABLE_ACCELERATION
        if self.speed_ms < self.target_speed_ms:
            self.speed_ms = min(self.speed_ms + accel * dt, self.target_speed_ms)
        elif self.speed_ms > self.target_speed_ms:
            self.speed_ms = max(self.speed_ms - accel * dt, 0.0)

    def advance_progress(self, distance: float, edge_length: float) -> None:
        """Advance the car's progress ratio along the edge.

        Args:
            distance: Distance travelled in metres this tick.
            edge_length: Total length of the current edge in metres.
        """
        if edge_length <= 0:
            return
        self.progress += distance / edge_length

    def switch_to_next_edge(self, edge: tuple[NodeID, NodeID, EdgeKey], lane: int) -> None:
        """Move to the next edge in the route.

        Args:
            edge: The new (u, v, key) edge.
            lane: The lane to occupy on the new edge.
        """
        self.current_edge = edge
        self.current_lane = lane
        self.progress = 0.0
        self.route_index += 1

    def assign_route(self, route: list[NodeID], start_edge: tuple[NodeID, NodeID, EdgeKey]) -> None:
        """Assign a new route and reset position.

        Args:
            route: Full list of node IDs (origin to destination).
            start_edge: The first edge to drive on.
        """
        self.route = route
        self.route_index = 0
        self.current_edge = start_edge
        self.progress = 0.0
        self.rerouting = False

    def reset_for_new_trip(
        self,
        route: list[NodeID],
        start_edge: tuple[NodeID, NodeID, EdgeKey],
        lane: int,
        spawn_time: float,
    ) -> None:
        """Reset the car for a new trip from origin to destination.

        Args:
            route: Full list of node IDs (origin to destination).
            start_edge: The first edge to drive on.
            lane: Lane to start on.
            spawn_time: New spawn time for this trip.
        """
        self.route = route
        self.route_index = 0
        self.current_edge = start_edge
        self.current_lane = lane
        self.progress = 0.0
        self.spawn_time = spawn_time
        self.speed_ms = config.DEFAULT_SPEED_MS
        self.target_speed_ms = config.DEFAULT_SPEED_MS
        self.waiting_at_light = False
        self.light_wait_seconds = 0.0
        self.waiting_for_car_ahead = False
        self.car_ahead_wait_seconds = 0.0
        self.rerouting = False

    def get_position_on_edge(
        self,
        geometry: LineString,
        lane_count: int,
    ) -> tuple[float, float]:
        """Compute the (x, y) pixel position of the car on its edge.

        Accounts for lane offset: cars in different lanes are visually
        shifted perpendicular to the road direction.

        Args:
            geometry: Shapely LineString of the road edge.
            lane_count: Total number of lanes on this edge.

        Returns:
            (x, y) pixel coordinates.
        """
        if geometry is None or geometry.length == 0:
            return (0.0, 0.0)

        # Clamp progress
        p = max(0.0, min(1.0, self.progress))
        point = geometry.interpolate(p, normalized=True)

        # Compute lane offset perpendicular to road direction
        frac = min(p + 0.001, 1.0)
        next_point = geometry.interpolate(frac, normalized=True)

        dx = next_point.x - point.x
        dy = next_point.y - point.y
        length = math.hypot(dx, dy)
        if length == 0:
            return (point.x, point.y)

        # Perpendicular direction (normalised)
        perp_x = -dy / length
        perp_y = dx / length

        # Lane offset: centre of the lane group
        lane_width = config.LANE_WIDTH_PX
        total_width = lane_count * lane_width
        # Offset from road centre to this lane's centre
        # Rightmost lane = most negative offset (driving on right)
        lane_offset = (self.current_lane - (lane_count - 1) / 2) * lane_width

        return (
            point.x + perp_x * lane_offset,
            point.y + perp_y * lane_offset,
        )

    def __repr__(self) -> str:
        return (
            f"Car({self.id}, edge={self.current_edge}, "
            f"lane={self.current_lane}, progress={self.progress:.2f}, "
            f"speed={self.speed_ms:.1f} m/s)"
        )