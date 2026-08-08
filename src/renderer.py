"""Pygame renderer — draws the road network, cars, traffic lights, and UI.

Coordinates are converted from geographic (lon/lat) to screen space using
the simulation's bounding box. The renderer is stateless for the drawing
pass — it receives the simulation state each frame.
"""

import logging
import math
from typing import Optional

import pygame
import numpy as np
from shapely.geometry import LineString

import config
from src.road_network import RoadNetwork, NodeID, EdgeKey
from src.traffic_simulation import TrafficSimulation
from src.traffic_light import TrafficLight, SignalPhase

log = logging.getLogger(__name__)


class Renderer:
    """Draws the simulation state to a Pygame surface.

    Attributes:
        screen: The main Pygame display surface.
        font: Default font for UI text.
        font_small: Small font for labels.
        font_large: Large font for titles.
        clock: Pygame clock for FPS tracking.
        bounds: (min_lon, min_lat, max_lon, max_lat) cached for performance.
        padding: Padding around the map in pixels.
        camera_x, camera_y: Camera pan offset in screen pixels.
        camera_zoom: Zoom factor (1.0 = default fit).
        _edge_cache: Cached screen coordinates for road edges.
        _node_cache: Cached screen coordinates for nodes.
        _road_network: Reference to the road network for node lookups.
    """

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.map_width = self.width - config.UI_PANEL_WIDTH
        self.map_height = self.height
        self.padding = 40

        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", config.UI_FONT_SIZE)
        self.font_small = pygame.font.SysFont("consolas", config.UI_FONT_SIZE_SMALL)
        self.font_large = pygame.font.SysFont("consolas", config.UI_FONT_SIZE_LARGE)
        self.clock = pygame.time.Clock()

        self.bounds: Optional[tuple[float, float, float, float]] = None
        self._edge_cache: dict[tuple[NodeID, NodeID, EdgeKey], list[tuple[float, float]]] = {}
        self._node_cache: dict[NodeID, tuple[float, float]] = {}
        self._hovered_edge: Optional[tuple[NodeID, NodeID, EdgeKey]] = None
        self._road_network: Optional[RoadNetwork] = None

        # Camera state
        self.camera_x: float = 0.0
        self.camera_y: float = 0.0
        self.camera_zoom: float = 1.0

        # Surface for the map area (to avoid redrawing static elements)
        self._map_surface: Optional[pygame.Surface] = None
        self._map_dirty = True

        # Track dragging state
        self._dragging: bool = False
        self._drag_start_x: float = 0.0
        self._drag_start_y: float = 0.0
        self._drag_cam_x: float = 0.0
        self._drag_cam_y: float = 0.0

    def set_bounds(self, bounds: tuple[float, float, float, float]) -> None:
        """Set the geographic bounds and invalidate caches."""
        self.bounds = bounds
        self._edge_cache.clear()
        self._node_cache.clear()
        self._map_dirty = True
        # Reset camera on new bounds
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_zoom = 1.0

    def set_road_network(self, rn: RoadNetwork) -> None:
        """Store a reference to the road network for node lookups."""
        self._road_network = rn

    # ── Camera controls ─────────────────────────────────────────

    def zoom_in(self) -> None:
        """Zoom in by a factor of 1.2."""
        self.camera_zoom *= 1.2
        self.camera_zoom = min(self.camera_zoom, 50.0)
        self._invalidate_cache()

    def zoom_out(self) -> None:
        """Zoom out by a factor of 1.2."""
        self.camera_zoom /= 1.2
        self.camera_zoom = max(self.camera_zoom, 0.1)
        self._invalidate_cache()

    def pan(self, dx: float, dy: float) -> None:
        """Pan the camera by *dx*, *dy* screen pixels."""
        self.camera_x += dx
        self.camera_y += dy
        self._invalidate_cache()

    def start_drag(self, x: float, y: float) -> None:
        """Begin a camera drag operation."""
        self._dragging = True
        self._drag_start_x = x
        self._drag_start_y = y
        self._drag_cam_x = self.camera_x
        self._drag_cam_y = self.camera_y

    def continue_drag(self, x: float, y: float) -> None:
        """Continue dragging the camera."""
        if self._dragging:
            self.camera_x = self._drag_cam_x + (x - self._drag_start_x)
            self.camera_y = self._drag_cam_y + (y - self._drag_start_y)
            self._invalidate_cache()

    def end_drag(self) -> None:
        """End a camera drag operation."""
        self._dragging = False

    def reset_view(self) -> None:
        """Reset camera to default (fit all)."""
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_zoom = 1.0
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Clear all cached screen coordinates so they recompute with new camera."""
        self._edge_cache.clear()
        self._node_cache.clear()
        self._map_dirty = True

    def render(self, simulation: TrafficSimulation) -> None:
        """Render one frame of the simulation.

        Args:
            simulation: The current simulation state.
        """
        # Refresh dimensions from the (possibly resized) screen
        self.width, self.height = self.screen.get_size()
        self.map_width = self.width - config.UI_PANEL_WIDTH
        self.map_height = self.height

        if self.bounds is None:
            self.set_bounds(simulation.get_bounds())

        # Clear screen
        self.screen.fill(config.hex_to_rgb(config.UI_BG_COLOR))

        # Draw map area background
        map_rect = pygame.Rect(0, 0, self.map_width, self.map_height)
        pygame.draw.rect(self.screen, (30, 30, 50), map_rect)

        # Draw roads
        self._render_roads(simulation)

        # Draw traffic lights
        self._render_traffic_lights(simulation)

        # Draw cars
        self._render_cars(simulation)

        # Draw UI panel
        self._render_ui_panel(simulation)

        # Update display
        pygame.display.flip()

    # ── Coordinate conversion ───────────────────────────────────

    def _to_screen(self, lon: float, lat: float) -> tuple[float, float]:
        """Convert geographic to screen coordinates, applying camera transform."""
        if self.bounds is None:
            return (0.0, 0.0)

        min_lon, min_lat, max_lon, max_lat = self.bounds
        draw_w = self.map_width - 2 * self.padding
        draw_h = self.map_height - 2 * self.padding

        if (max_lon - min_lon) <= 0 or (max_lat - min_lat) <= 0:
            return (self.padding, self.padding)

        # Base scale (fit to screen)
        map_aspect = (max_lon - min_lon) / (max_lat - min_lat)
        screen_aspect = draw_w / draw_h

        if map_aspect > screen_aspect:
            base_scale = draw_w / (max_lon - min_lon)
            offset_x = self.padding
            offset_y = self.padding + (draw_h - (max_lat - min_lat) * base_scale) / 2
        else:
            base_scale = draw_h / (max_lat - min_lat)
            offset_y = self.padding
            offset_x = self.padding + (draw_w - (max_lon - min_lon) * base_scale) / 2

        # Apply camera zoom: the camera zooms around the centre of the map
        centre_x = self.map_width / 2
        centre_y = self.map_height / 2

        # Base screen position
        base_x = offset_x + (lon - min_lon) * base_scale
        base_y = offset_y + (max_lat - lat) * base_scale  # invert Y

        # Apply zoom (relative to map centre) and pan
        zoomed_x = centre_x + (base_x - centre_x) * self.camera_zoom + self.camera_x
        zoomed_y = centre_y + (base_y - centre_y) * self.camera_zoom + self.camera_y

        return (zoomed_x, zoomed_y)

    # ── Road rendering ──────────────────────────────────────────

    def _render_roads(self, simulation: TrafficSimulation) -> None:
        """Draw all road edges onto the map surface.

        Uses two passes:
          Pass 1 — draw all non-blocked edges (solid).
          Pass 2 — draw blocked edges LAST (dashed red on top) so they are
                   visible even when parallel edges share the same geometry.
        """
        rn = simulation.road_network
        blocked = rn.blocked_edges

        # ── Pre-compute screen coordinates for all edges ──
        for u, v, key in rn.get_all_edges():
            if (u, v, key) not in self._edge_cache:
                edge_data = rn.get_edge_attributes(u, v, key)
                points = self._geometry_to_screen(edge_data.geometry, u, v)
                self._edge_cache[(u, v, key)] = points

        highlighted = self._hovered_edge

        # ── Pass 1: draw non-blocked edges (solid) ──
        for u, v, key in rn.get_all_edges():
            if (u, v, key) in blocked:
                continue  # handled in pass 2

            points = self._edge_cache.get((u, v, key))
            if not points or len(points) < 2:
                continue

            is_highlighted = highlighted == (u, v, key)
            if is_highlighted:
                color = config.hex_to_rgb(config.ROAD_HIGHLIGHT_COLOR)
                self._draw_road(points, color, width=3)
            else:
                edge_data = rn.get_edge_attributes(u, v, key)
                width = max(
                    config.ROAD_MIN_WIDTH_PX,
                    edge_data.lanes * config.LANE_WIDTH_PX,
                )
                color = config.hex_to_rgb(config.ROAD_COLOR)
                self._draw_road(points, color, width)

        # ── Pass 2: draw blocked edges on top (dashed red) ──
        for edge in blocked:
            points = self._edge_cache.get(edge)
            if not points or len(points) < 2:
                continue
            color = config.hex_to_rgb(config.ROAD_BLOCKED_COLOR)
            self._draw_road_dashed(points, color)

    def _geometry_to_screen(
        self,
        geometry: Optional[LineString],
        u: NodeID,
        v: NodeID,
    ) -> list[tuple[float, float]]:
        """Convert a shapely LineString to screen coordinates.

        Falls back to straight line between nodes if no geometry.
        """
        if geometry is not None and not geometry.is_empty:
            coords = list(geometry.coords)
            return [self._to_screen(lon, lat) for lon, lat in coords]

        # Fallback: straight line between nodes
        pos_u = self._get_node_screen(u)
        pos_v = self._get_node_screen(v)
        if pos_u is None or pos_v is None:
            return []
        return [pos_u, pos_v]

    def _get_node_screen(self, node_id: NodeID) -> Optional[tuple[float, float]]:
        """Get screen coordinates for a node (cached)."""
        if node_id in self._node_cache:
            return self._node_cache[node_id]
        if self.bounds is None or self._road_network is None:
            return None
        pos = self._road_network.get_node_position(node_id)
        if pos is None:
            return None
        screen_pos = self._to_screen(pos[0], pos[1])
        self._node_cache[node_id] = screen_pos
        return screen_pos

    def _draw_road(
        self,
        points: list[tuple[float, float]],
        color: tuple[int, int, int],
        width: int = 2,
    ) -> None:
        """Draw a solid road line."""
        if len(points) < 2:
            return
        # Draw as a series of line segments for smooth curves
        pygame.draw.lines(self.screen, color, False, points, width)

    def _draw_road_dashed(
        self,
        points: list[tuple[float, float]],
        color: tuple[int, int, int],
        width: int = 2,
        dash_length: int = 8,
        gap_length: int = 5,
    ) -> None:
        """Draw a dashed road line (for blocked roads)."""
        if len(points) < 2:
            return

        # Draw dashed segments
        total_length = 0.0
        seg_lengths = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            length = math.hypot(x2 - x1, y2 - y1)
            seg_lengths.append(length)
            total_length += length

        if total_length == 0:
            return

        # Walk along the polyline, drawing dashes
        pos = 0.0
        drawing = True
        dash_start = 0.0

        # Simple approach: draw dashed segments along the polyline
        # Convert to a flat list of segments
        segments = []
        for i in range(len(points) - 1):
            segments.append((points[i], points[i + 1]))

        # Draw dashed lines
        for seg in segments:
            (x1, y1), (x2, y2) = seg
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len == 0:
                continue

            # Number of dashes along this segment
            n_dashes = int(seg_len / (dash_length + gap_length))
            for d in range(n_dashes + 1):
                t_start = d * (dash_length + gap_length) / seg_len
                t_end = min(t_start + dash_length / seg_len, 1.0)
                if t_start >= 1.0:
                    break
                sx = x1 + (x2 - x1) * t_start
                sy = y1 + (y2 - y1) * t_start
                ex = x1 + (x2 - x1) * t_end
                ey = y1 + (y2 - y1) * t_end
                pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), width)

    # ── Traffic light rendering ─────────────────────────────────

    def _render_traffic_lights(self, simulation: TrafficSimulation) -> None:
        """Draw traffic light indicators at intersections."""
        for light in simulation.traffic_lights:
            lon, lat = light.position
            sx, sy = self._to_screen(lon, lat)

            # Only draw if on screen
            if 0 <= sx <= self.map_width and 0 <= sy <= self.map_height:
                color = light.get_color()
                radius = config.TRAFFIC_LIGHT_RADIUS_PX
                pygame.draw.circle(self.screen, color, (int(sx), int(sy)), radius)
                # Add a white border for visibility
                pygame.draw.circle(self.screen, (255, 255, 255), (int(sx), int(sy)), radius, 1)

    # ── Car rendering ───────────────────────────────────────────

    def _render_cars(self, simulation: TrafficSimulation) -> None:
        """Draw all cars as small oriented rectangles."""
        for car in simulation.cars:
            u, v, key = car.current_edge
            edge_data = simulation.road_network.get_edge_attributes(u, v, key)
            geometry = edge_data.geometry

            # Get car position on the edge
            points = self._edge_cache.get((u, v, key))
            if not points or len(points) < 2:
                continue

            # Interpolate position along the screen polyline
            car_pos = self._interpolate_along_polyline(points, car.progress)

            if car_pos is None:
                continue

            # Compute lane offset
            lane_count = max(edge_data.lanes, 1)
            lane_width = config.LANE_WIDTH_PX
            lane_offset = (car.current_lane - (lane_count - 1) / 2) * lane_width

            # Direction of the road at this point
            direction = self._get_direction_at_progress(points, car.progress)

            if direction is not None:
                # Perpendicular offset
                perp_x, perp_y = -direction[1], direction[0]
                off_x = perp_x * lane_offset
                off_y = perp_y * lane_offset
                cx = car_pos[0] + off_x
                cy = car_pos[1] + off_y
            else:
                cx, cy = car_pos

            # Draw car as a small rectangle oriented along the road
            color = config.hex_to_rgb(car.color)
            if direction is not None:
                # Draw outline (white) first, then filled car on top
                car_len = config.CAR_LENGTH_PX
                car_wid = config.CAR_WIDTH_PX
                angle_rad = math.atan2(direction[1], direction[0])
                self._draw_oriented_rect(
                    self.screen, (255, 255, 255), (cx, cy),
                    car_len + 2, car_wid + 2, angle_rad,
                )
                self._draw_oriented_rect(
                    self.screen, color, (cx, cy),
                    car_len, car_wid, angle_rad,
                )
            else:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(cx), int(cy)), 5)
                pygame.draw.circle(self.screen, color, (int(cx), int(cy)), 4)

    def _interpolate_along_polyline(
        self,
        points: list[tuple[float, float]],
        progress: float,
    ) -> Optional[tuple[float, float]]:
        """Interpolate a position along a polyline by progress ratio [0, 1]."""
        if len(points) < 2:
            return None

        # Compute cumulative lengths
        lengths = [0.0]
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            lengths.append(lengths[-1] + math.hypot(dx, dy))

        total_length = lengths[-1]
        if total_length == 0:
            return points[0]

        target_dist = progress * total_length

        # Find the segment containing this distance
        for i in range(len(points) - 1):
            seg_start = lengths[i]
            seg_end = lengths[i + 1]
            if seg_start <= target_dist <= seg_end:
                seg_len = seg_end - seg_start
                if seg_len == 0:
                    return points[i]
                t = (target_dist - seg_start) / seg_len
                x = points[i][0] + (points[i + 1][0] - points[i][0]) * t
                y = points[i][1] + (points[i + 1][1] - points[i][1]) * t
                return (x, y)

        return points[-1]

    def _get_direction_at_progress(
        self,
        points: list[tuple[float, float]],
        progress: float,
    ) -> Optional[tuple[float, float]]:
        """Get the direction vector at a given progress along the polyline."""
        if len(points) < 2:
            return None

        lengths = [0.0]
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            lengths.append(lengths[-1] + math.hypot(dx, dy))

        total_length = lengths[-1]
        if total_length == 0:
            return None

        target_dist = progress * total_length

        for i in range(len(points) - 1):
            seg_start = lengths[i]
            seg_end = lengths[i + 1]
            if seg_start <= target_dist <= seg_end:
                dx = points[i + 1][0] - points[i][0]
                dy = points[i + 1][1] - points[i][1]
                seg_len = math.hypot(dx, dy)
                if seg_len == 0:
                    continue
                return (dx / seg_len, dy / seg_len)

        # Return direction of last segment
        dx = points[-1][0] - points[-2][0]
        dy = points[-1][1] - points[-2][1]
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            return None
        return (dx / seg_len, dy / seg_len)

    def _draw_oriented_rect(
        self,
        surface: pygame.Surface,
        color: tuple[int, int, int],
        center: tuple[float, float],
        length: float,
        width: float,
        angle: float,
    ) -> None:
        """Draw a rectangle centred on *center* with given orientation."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Half-dimensions
        hw = width / 2
        hl = length / 2

        # Four corners (local coords)
        corners = [
            (hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw),
        ]

        # Rotate and translate
        points = []
        for lx, ly in corners:
            rx = center[0] + lx * cos_a - ly * sin_a
            ry = center[1] + lx * sin_a + ly * cos_a
            points.append((int(rx), int(ry)))

        pygame.draw.polygon(surface, color, points)

    # ── UI Panel ─────────────────────────────────────────────────

    def _render_ui_panel(self, simulation: TrafficSimulation) -> None:
        """Draw the right-side UI control panel."""
        panel_x = self.map_width
        panel_w = config.UI_PANEL_WIDTH
        bg_color = config.hex_to_rgb(config.UI_BG_COLOR)

        # Background
        panel_rect = pygame.Rect(panel_x, 0, panel_w, self.height)
        pygame.draw.rect(self.screen, bg_color, panel_rect)
        pygame.draw.line(
            self.screen, (100, 100, 100),
            (panel_x, 0), (panel_x, self.height), 1,
        )

        # Title
        title = self.font_large.render("TRAFFIC SIM", True, (200, 200, 200))
        self.screen.blit(title, (panel_x + 10, 10))

        # GA optimisation label (if active)
        if simulation.ga_label:
            ga_text = self.font_small.render(
                simulation.ga_label, True, (100, 255, 100),  # green highlight
            )
            self.screen.blit(ga_text, (panel_x + 10, 35))
            y_offset = 60
        else:
            y_offset = 40

        # City name
        city_text = self.font_small.render(
            f"City: {simulation.road_network.node_count} nodes",
            True, (150, 150, 150),
        )
        self.screen.blit(city_text, (panel_x + 10, y_offset))

        # ── Car count and spawn rate ──
        y = y_offset + 25
        label = self.font.render(f"Cars: {len(simulation.cars)}", True, (200, 200, 200))
        self.screen.blit(label, (panel_x + 10, y))
        y += 25
        rate_text = self.font_small.render(f"Spawn rate: {simulation.poisson_rate:.2f} cars/s", True, (150, 150, 150))
        self.screen.blit(rate_text, (panel_x + 10, y))
        y += 25

        # ── Speed multiplier ──
        speed_text = self.font.render(f"Speed: {simulation.speed_multiplier:.2f}x", True, (200, 200, 200))
        self.screen.blit(speed_text, (panel_x + 10, y))
        y += 30

        # ── Algorithm selector ──
        algo_label = self.font.render(f"Algorithm: {simulation.algorithm}", True, (200, 200, 200))
        self.screen.blit(algo_label, (panel_x + 10, y))
        y += 30

        for algo in config.AVAILABLE_ALGORITHMS:
            algo_color = (100, 200, 100) if algo == simulation.algorithm else (150, 150, 150)
            algo_text = self.font_small.render(f"  {algo}", True, algo_color)
            self.screen.blit(algo_text, (panel_x + 20, y))
            y += 20

        y += 10

        # ── Blocked edges ──
        blocked_count = len(simulation.road_network.blocked_edges)
        blocked_label = self.font.render(f"Blocked: {blocked_count}", True, (200, 100, 100))
        self.screen.blit(blocked_label, (panel_x + 10, y))
        y += 30

        # ── Stats ──
        stats = simulation.get_stats()
        stats_label = self.font.render("Stats:", True, (200, 200, 200))
        self.screen.blit(stats_label, (panel_x + 10, y))
        y += 25

        for key, value in stats.items():
            if key in ("total_travel_time_seconds", "completed_trips"):
                continue  # skip raw values, show average instead
            stat_text = self.font_small.render(f"  {key}: {value}", True, (150, 150, 150))
            self.screen.blit(stat_text, (panel_x + 10, y))
            y += 18

        # ── Average travel time (highlighted) ──
        avg_time = stats.get("avg_travel_time_seconds", 0.0)
        if stats.get("completed_trips", 0) > 0:
            avg_text = self.font_small.render(
                f"  Avg travel time: {avg_time:.1f}s",
                True, (100, 200, 255),  # light blue highlight
            )
        else:
            avg_text = self.font_small.render(
                "  Avg travel time: -- (no trips completed)",
                True, (120, 120, 120),
            )
        self.screen.blit(avg_text, (panel_x + 10, y))
        y += 18

        y += 10

        # ── Controls help ──
        controls = [
            "Controls:",
            "↑/↓: +/- 10 cars",
            "←/→: +/- 1 car",
            "1-5: Algorithm",
            "[: Slow down (0.25x)",
            "]: Speed up (0.25x)",
            "Scroll: Zoom in/out",
            "MClick-drag: Pan",
            "Click road: Block",
            "R: Reset view",
            "ESC: Exit",
        ]
        for line in controls:
            ctrl_text = self.font_small.render(line, True, (120, 120, 120))
            self.screen.blit(ctrl_text, (panel_x + 10, y))
            y += 18

    # ── Hit testing ─────────────────────────────────────────────

    def find_edge_at_screen(
        self,
        screen_x: float,
        screen_y: float,
        simulation: TrafficSimulation,
        threshold: float = 8.0,
        prefer_reverse: bool = False,
    ) -> Optional[tuple[NodeID, NodeID, EdgeKey]]:
        """Find the nearest road edge to a screen point.

        For two-way roads (both (u,v) and (v,u) exist), the function
        returns the forward direction (u,v) by default, or the reverse
        direction (v,u) when prefer_reverse=True. This allows left-click
        to block one direction and right-click to block the other.

        Args:
            screen_x, screen_y: Screen pixel coordinates.
            simulation: The simulation state.
            threshold: Max distance in pixels to consider a hit.
            prefer_reverse: If True, prefer the reverse direction for
                two-way roads.

        Returns:
            (u, v, key) of the nearest edge, or None.
        """
        best_edge = None
        best_dist = float("inf")
        rn = simulation.road_network

        for u, v, key in rn.get_all_edges():
            # Try cache first, compute on-the-fly if missing
            points = self._edge_cache.get((u, v, key))
            if not points or len(points) < 2:
                points = self._geometry_to_screen(
                    rn.get_edge_attributes(u, v, key).geometry, u, v,
                )
                if not points or len(points) < 2:
                    continue

            # Compute minimum distance from click point to the polyline
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                dist = self._point_to_segment_distance(
                    screen_x, screen_y, x1, y1, x2, y2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_edge = (u, v, key)

        if best_dist <= threshold:
            # For two-way roads, prefer the requested direction
            if best_edge is not None and prefer_reverse:
                u, v, key = best_edge
                if rn.get_edge(v, u, key) is not None:
                    best_edge = (v, u, key)
            self._hovered_edge = best_edge
            return best_edge

        self._hovered_edge = None
        return None

    @staticmethod
    def _point_to_segment_distance(
        px: float, py: float,
        x1: float, y1: float,
        x2: float, y2: float,
    ) -> float:
        """Compute the minimum distance from point P to segment AB."""
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)

        # Project point onto the line, clamped to [0, 1]
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return math.hypot(px - closest_x, py - closest_y)