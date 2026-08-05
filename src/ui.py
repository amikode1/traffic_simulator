"""UI input handling — processes keyboard and mouse events.

Separates input logic from rendering and simulation. Translates user
actions (key presses, mouse clicks, scroll wheel) into simulation
and camera method calls.
"""

import logging
from typing import Optional, Callable

import pygame

import config
from src.road_network import NodeID, EdgeKey
from src.traffic_simulation import TrafficSimulation

log = logging.getLogger(__name__)


class UIHandler:
    """Processes user input and delegates to the simulation and renderer.

    Pure input handler — no rendering logic. Receives Pygame events
    and calls methods on the simulation and renderer.

    Attributes:
        car_count: The current car count target (for slider feedback).
        selected_algorithm: Currently selected algorithm index.
        mouse_pos: Current mouse position (screen coords).
        mouse_on_map: Whether the mouse is over the map area.
        left_click: Whether left mouse button was just pressed this frame.
        zoom_fn: Callable(amount) for zooming the camera.
        pan_fn: Callable(dx, dy) for panning the camera.
        reset_view_fn: Callable() to reset camera.
        start_drag_fn: Callable(x, y) to begin camera drag.
        continue_drag_fn: Callable(x, y) to continue camera drag.
        end_drag_fn: Callable() to end camera drag.
        _is_dragging: Whether we are currently in a camera drag.
    """

    def __init__(self):
        self.car_count: int = config.DEFAULT_CAR_COUNT
        self.selected_algorithm: int = 0
        self.mouse_pos: tuple[int, int] = (0, 0)
        self.mouse_on_map: bool = False
        self.left_click: bool = False
        self.keys_pressed: set[int] = set()
        self.screen_width: int = config.WINDOW_WIDTH
        self.screen_height: int = config.WINDOW_HEIGHT

        # Camera control callbacks (set by main.py)
        self.zoom_fn: Optional[Callable[[float], None]] = None
        self.pan_fn: Optional[Callable[[float, float], None]] = None
        self.reset_view_fn: Optional[Callable[[], None]] = None
        self.start_drag_fn: Optional[Callable[[float, float], None]] = None
        self.continue_drag_fn: Optional[Callable[[float, float], None]] = None
        self.end_drag_fn: Optional[Callable[[], None]] = None
        self.resize_fn: Optional[Callable[[int, int], None]] = None

        self._is_dragging: bool = False
        self._last_mouse_pos: tuple[int, int] = (0, 0)

    def process_events(self, simulation: TrafficSimulation) -> bool:
        """Process all pending Pygame events.

        Args:
            simulation: The simulation to modify.

        Returns:
            False if the user requested to quit.
        """
        running = True
        self.left_click = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
                running = self._handle_keydown(event.key, simulation)

            elif event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)

            elif event.type == pygame.VIDEORESIZE:
                # Window was resized — update dimensions and notify main
                self.screen_width = event.w
                self.screen_height = event.h
                if self.resize_fn:
                    self.resize_fn(event.w, event.h)

            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
                self.mouse_on_map = event.pos[0] < (self.screen_width - config.UI_PANEL_WIDTH)

                # Handle drag
                if self._is_dragging and self.continue_drag_fn:
                    self.continue_drag_fn(event.pos[0], event.pos[1])

                self._last_mouse_pos = event.pos

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_pos = event.pos
                self.mouse_on_map = event.pos[0] < (self.screen_width - config.UI_PANEL_WIDTH)

                # Scroll wheel for zoom
                if event.button == 4:  # scroll up
                    if self.mouse_on_map and self.zoom_fn:
                        self.zoom_fn(1)  # zoom in
                elif event.button == 5:  # scroll down
                    if self.mouse_on_map and self.zoom_fn:
                        self.zoom_fn(-1)  # zoom out
                # Middle mouse button for drag-pan
                elif event.button == 2:  # middle click
                    if self.mouse_on_map and self.start_drag_fn:
                        self._is_dragging = True
                        self.start_drag_fn(event.pos[0], event.pos[1])
                # Left click
                elif event.button == 1:
                    self.left_click = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2 and self._is_dragging:
                    self._is_dragging = False
                    if self.end_drag_fn:
                        self.end_drag_fn()

        return running

    def _handle_keydown(self, key: int, simulation: TrafficSimulation) -> bool:
        """Handle a single key press. Returns False if quit requested."""
        if key == pygame.K_ESCAPE:
            return False

        # ── Car count control ──
        elif key == pygame.K_UP:
            self.car_count = min(self.car_count + 10, config.MAX_CAR_COUNT)
            simulation.set_desired_car_count(self.car_count)
        elif key == pygame.K_DOWN:
            self.car_count = max(self.car_count - 10, config.MIN_CAR_COUNT)
            simulation.set_desired_car_count(self.car_count)
        elif key == pygame.K_RIGHT:
            self.car_count = min(self.car_count + 1, config.MAX_CAR_COUNT)
            simulation.set_desired_car_count(self.car_count)
        elif key == pygame.K_LEFT:
            self.car_count = max(self.car_count - 1, config.MIN_CAR_COUNT)
            simulation.set_desired_car_count(self.car_count)

        # ── Algorithm selection ──
        elif key == pygame.K_1:
            self._set_algorithm(simulation, 0)
        elif key == pygame.K_2:
            self._set_algorithm(simulation, 1)
        elif key == pygame.K_3:
            self._set_algorithm(simulation, 2)
        elif key == pygame.K_4:
            self._set_algorithm(simulation, 3)
        elif key == pygame.K_5:
            self._set_algorithm(simulation, 4)

        # ── Camera controls ──
        elif key == pygame.K_r and self.reset_view_fn:
            self.reset_view_fn()
            log.info("View reset")
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            if self.zoom_fn:
                self.zoom_fn(1)
        elif key == pygame.K_MINUS:
            if self.zoom_fn:
                self.zoom_fn(-1)

        # ── Speed control ──
        elif key == pygame.K_LEFTBRACKET:
            self._adjust_speed(simulation, -1)
        elif key == pygame.K_RIGHTBRACKET:
            self._adjust_speed(simulation, 1)

        return True

    def _set_algorithm(self, simulation: TrafficSimulation, index: int) -> None:
        """Change the pathfinding algorithm by index."""
        if 0 <= index < len(config.AVAILABLE_ALGORITHMS):
            self.selected_algorithm = index
            algo = config.AVAILABLE_ALGORITHMS[index]
            simulation.set_algorithm(algo)
            log.info("Algorithm changed to: %s", algo)

    def _adjust_speed(self, simulation: TrafficSimulation, direction: int) -> None:
        """Increase or decrease the simulation speed multiplier.

        Args:
            simulation: The simulation to modify.
            direction: +1 to speed up, -1 to slow down.
        """
        step = config.SPEED_MULTIPLIER_STEP * direction
        new_speed = simulation.speed_multiplier + step
        simulation.set_speed_multiplier(new_speed)
        log.info("Speed multiplier: %.2fx", simulation.speed_multiplier)

    def handle_click_on_map(
        self,
        simulation: TrafficSimulation,
        find_edge_fn: Callable,
    ) -> None:
        """Handle a mouse click on the map area (block/unblock road).

        Args:
            simulation: The simulation to modify.
            find_edge_fn: A callable that takes (x, y, simulation) and returns
                an edge tuple or None.
        """
        if not self.left_click or not self.mouse_on_map:
            return

        x, y = self.mouse_pos
        edge = find_edge_fn(x, y, simulation)
        if edge is not None:
            u, v, key = edge
            is_blocked = simulation.toggle_block_edge(u, v, key)
            log.info("Edge (%d, %d, %d) %s", u, v, key,
                     "blocked" if is_blocked else "unblocked")