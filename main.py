"""Traffic Simulator — Main Entry Point.

Accepts optional --ga <file> flag to load GA-optimised road closures:
    python main.py "Manhattan" --ga data/ga_result_Manhattan__New_York__USA.json
"""

import json
import logging
import math
import os
import sys
import threading
import time
from typing import Optional

import pygame

import config
from src.map_manager import load_city
from src.traffic_simulation import TrafficSimulation
from src.renderer import Renderer
from src.ui import UIHandler

# ── Logging setup ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Loading screen ──────────────────────────────────────────────

def _draw_loading_screen(
    screen: pygame.Surface,
    city_name: str,
    dot_count: int,
    spinner_angle: float,
) -> None:
    """Draw a loading animation while the map downloads.

    Args:
        screen: The Pygame display surface.
        city_name: Human-readable city name being loaded.
        dot_count: Number of trailing dots (0-3) for the animation.
        spinner_angle: Current angle of the rotating spinner in radians.
    """
    w, h = screen.get_size()
    bg_color = config.hex_to_rgb(config.UI_BG_COLOR)
    screen.fill(bg_color)

    font_large = pygame.font.SysFont("consolas", 28)
    font_small = pygame.font.SysFont("consolas", 16)

    # ── Title ──
    dots = "." * dot_count
    title = font_large.render(f"Loading {city_name}{dots}", True, (200, 200, 200))
    title_rect = title.get_rect(center=(w // 2, h // 2 - 60))
    screen.blit(title, title_rect)

    # ── Hint ──
    hint = font_small.render(
        "Cached maps load instantly; new maps download from OpenStreetMap.", True,
        (120, 120, 120),
    )
    hint_rect = hint.get_rect(center=(w // 2, h // 2 - 20))
    screen.blit(hint, hint_rect)

    # ── Spinning arc ──
    cx, cy = w // 2, h // 2 + 40
    radius = 16
    # Draw a dashed circle as a spinner: 12 segments, one highlighted
    n_segments = 12
    for i in range(n_segments):
        theta0 = spinner_angle + (2 * math.pi * i / n_segments)
        theta1 = spinner_angle + (2 * math.pi * (i + 2) / n_segments)  # 2-segment arc
        x0 = cx + radius * math.cos(theta0)
        y0 = cy + radius * math.sin(theta0)
        x1 = cx + radius * math.cos(theta1)
        y1 = cy + radius * math.sin(theta1)
        # Brightness fades along the arc
        brightness = 60 + 180 * (n_segments - i) // n_segments
        color = (brightness, brightness, brightness)
        pygame.draw.line(screen, color, (x0, y0), (x1, y1), 4)

    # ── "Press ESC to cancel" ──
    esc_text = font_small.render("Press ESC to cancel", True, (80, 80, 80))
    esc_rect = esc_text.get_rect(center=(w // 2, h // 2 + 90))
    screen.blit(esc_text, esc_rect)

    pygame.display.flip()


def _show_error_screen(screen: pygame.Surface, message: str) -> None:
    """Display a fatal error message and wait a few seconds."""
    w, h = screen.get_size()
    bg_color = config.hex_to_rgb(config.UI_BG_COLOR)
    screen.fill(bg_color)
    font = pygame.font.SysFont("consolas", 22)
    text = font.render(f"Error: {message}", True, (255, 80, 80))
    text_rect = text.get_rect(center=(w // 2, h // 2 - 10))
    screen.blit(text, text_rect)
    sub = pygame.font.SysFont("consolas", 16).render(
        "Check city name or network connection, then restart.", True, (150, 150, 150),
    )
    sub_rect = sub.get_rect(center=(w // 2, h // 2 + 20))
    screen.blit(sub, sub_rect)
    pygame.display.flip()
    pygame.time.wait(5000)


# ── Main ────────────────────────────────────────────────────────

def main() -> None:
    """Run the traffic simulator application.

    Usage:
        python main.py "Manhattan"
        python main.py "Greenwood_Township" --ga data/ga_result_Greenwood_Township.json
    """
    # ── Parse optional command-line args ──
    city_name = config.DEFAULT_CITY
    ga_result_path: Optional[str] = None
    args = sys.argv[1:]
    if "--ga" in args:
        idx = args.index("--ga")
        if idx + 1 < len(args):
            ga_result_path = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
    if args:
        city_name = " ".join(args)

    # ── Initialise Pygame ──
    pygame.init()
    screen = pygame.display.set_mode(
        (config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
        pygame.DOUBLEBUF | pygame.RESIZABLE,
    )
    pygame.display.set_caption("Traffic Simulator")
    clock = pygame.time.Clock()

    # ── Load the city (background thread with loading animation) ──
    log.info("Loading city: %s", city_name)

    result_holder: list = []  # thread writes (road_network, traffic_lights) or raises
    error_holder: list = []

    def _load_thread() -> None:
        try:
            rn, tl = load_city(city_name)
            result_holder.append((rn, tl))
        except ValueError as exc:
            error_holder.append(exc)

    load_thread = threading.Thread(target=_load_thread, daemon=True)
    load_thread.start()

    # Loading animation loop
    dot_counter = 0
    spinner_angle = 0.0
    loading_cancelled = False

    while load_thread.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                loading_cancelled = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                loading_cancelled = True

        if loading_cancelled:
            # The daemon thread will be killed when the process exits
            pygame.quit()
            log.info("Loading cancelled by user.")
            return

        # Animate: cycle dots every 8 frames, rotate spinner
        dot_counter = (dot_counter + 1) % 24
        dot_count = (dot_counter // 6) % 4  # 0, 1, 2, 3 repeating
        spinner_angle += 0.15

        _draw_loading_screen(screen, city_name, dot_count, spinner_angle)
        clock.tick(30)  # 30 fps for the loading screen

    # Thread finished — check for errors
    if error_holder:
        exc = error_holder[0]
        log.error("Failed to load city: %s", exc)
        _show_error_screen(screen, str(exc))
        pygame.quit()
        return

    road_network, traffic_lights = result_holder[0]

    log.info(
        "Road network: %d nodes, %d edges, %d traffic lights",
        road_network.node_count,
        road_network.edge_count,
        len(traffic_lights),
    )

    # ── Apply GA-optimised road closures if requested ──
    if ga_result_path is not None:
        if not os.path.exists(ga_result_path):
            log.error("GA result file not found: %s", ga_result_path)
            _show_error_screen(screen, f"GA result not found: {ga_result_path}")
            pygame.quit()
            return

        try:
            with open(ga_result_path) as f:
                ga_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.error("Failed to read GA result: %s", exc)
            _show_error_screen(screen, f"Invalid GA result file: {exc}")
            pygame.quit()
            return

        closed_roads = ga_data.get("closed_roads", [])
        blocked_count = 0
        for road in closed_roads:
            if road.get("close", False):
                u, v, key = road["edge"]
                if road_network.block_edge(u, v, key):
                    blocked_count += 1
                # NOTE: The GA only tested closing this one direction (u→v).
                # We do NOT block the opposite direction to stay consistent
                # with what the GA evaluated.

        benefit = ga_data.get("benefit", 0.0)
        optimal_avg = ga_data.get("optimal_avg", 0.0)
        log.info(
            "Applied GA-optimised solution: %d roads blocked, "
            "benefit=%.3fs, optimal avg=%.2fs",
            blocked_count, benefit, optimal_avg,
        )

        # Store GA info for later use after simulation creation
        _ga_label_text = (
            f"GA Optimised: {blocked_count} roads closed, "
            f"benefit {benefit:.2f}s"
        )
        _ga_window_title = (
            f"Traffic Simulator — GA Optimised "
            f"({blocked_count} roads closed, benefit {benefit:.2f}s)"
        )
    else:
        _ga_label_text = ""
        _ga_window_title = None

    # ── Create simulation ──
    simulation = TrafficSimulation(
        road_network=road_network,
        traffic_lights=traffic_lights,
        desired_car_count=config.DEFAULT_CAR_COUNT,
    )
    # Warm-up period: discard trips completed before this simulation time
    # so the average travel time reflects steady-state congestion, not
    # the early empty-road period.
    simulation.set_warmup(60.0)

    # Apply GA info to simulation and window title (after simulation exists)
    if _ga_label_text:
        simulation.ga_label = _ga_label_text
        if _ga_window_title:
            pygame.display.set_caption(_ga_window_title)

    # ── Create renderer & UI ──
    renderer = Renderer(screen)
    renderer.set_bounds(simulation.get_bounds())
    renderer.set_road_network(road_network)
    ui_handler = UIHandler()

    # ── Wire up camera controls ──
    def zoom_camera(direction: float) -> None:
        if direction > 0:
            renderer.zoom_in()
        else:
            renderer.zoom_out()

    ui_handler.zoom_fn = zoom_camera
    ui_handler.pan_fn = renderer.pan
    ui_handler.reset_view_fn = renderer.reset_view
    ui_handler.start_drag_fn = renderer.start_drag
    ui_handler.continue_drag_fn = renderer.continue_drag
    ui_handler.end_drag_fn = renderer.end_drag

    def _resize_window(w: int, h: int) -> None:
        """Handle window resize — update the Pygame display mode."""
        pygame.display.set_mode((w, h), pygame.DOUBLEBUF | pygame.RESIZABLE)
        # Bounds and caches will be refreshed on next render() call

    ui_handler.resize_fn = _resize_window

    # ── Main loop ──
    running = True
    fixed_dt = 1.0 / config.TICK_RATE_HZ

    # Pre-warm: simulate one tick immediately so cars start spawning right away
    simulation.update(fixed_dt * simulation.speed_multiplier)

    while running:
        # ── Process input ──
        running = ui_handler.process_events(simulation)

        # ── Handle click on map → block/unblock road ──
        if (ui_handler.left_click or ui_handler.right_click) and ui_handler.mouse_on_map:
            ui_handler.handle_click_on_map(
                simulation,
                renderer.find_edge_at_screen,
            )

        # ── Update simulation (fixed timestep) ──
        # clock.tick(FPS) returns milliseconds since last call — this drives the loop
        raw_dt = clock.tick(config.TICK_RATE_HZ) / 1000.0
        raw_dt = min(raw_dt, 0.05)  # cap to 50 ms to prevent spiral of death

        # Apply speed multiplier — dt_scaled is fed to the simulation
        dt_scaled = fixed_dt * simulation.speed_multiplier
        simulation.update(dt_scaled)

        # ── Render frame ──
        renderer.render(simulation)

    # ── Clean shutdown ──
    pygame.quit()
    log.info("Simulation ended.")


if __name__ == "__main__":
    main()