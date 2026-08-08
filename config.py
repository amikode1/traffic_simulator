"""Configuration constants for the Traffic Simulator.

All tunable parameters live here. No magic numbers anywhere in the codebase.
"""

from typing import Final

# ── Simulation ────────────────
DEFAULT_CAR_COUNT: Final[int] = 300
MAX_CAR_COUNT: Final[int] = 1500
MIN_CAR_COUNT: Final[int] = 0
TICK_RATE_HZ: Final[int] = 60
SPAWN_INTERVAL_SECONDS: Final[float] = 0.5
POISSON_SPAWN_RATE: Final[float] = 0.5  # cars per second (λ) — light traffic

# ── Simulation speed ──────────
MIN_SPEED_MULTIPLIER: Final[float] = 0.1
MAX_SPEED_MULTIPLIER: Final[float] = 5.0
DEFAULT_SPEED_MULTIPLIER: Final[float] = 1.0
SPEED_MULTIPLIER_STEP: Final[float] = 0.25

# ── Car defaults ──────────────
DEFAULT_SPEED_KPH: Final[float] = 80.0  # km/h (~50 mph, city arterial speed)
DEFAULT_SPEED_MS: Final[float] = DEFAULT_SPEED_KPH / 3.6  # m/s
CAR_LENGTH_PX: Final[int] = 10
CAR_WIDTH_PX: Final[int] = 6
CAR_COLOR: Final[str] = "#f39c12"  # bright orange
CAR_HIGHLIGHT_COLOR: Final[str] = "#e74c3c"  # red (for rerouting)

# ── Car-following / queueing ───
CAR_LENGTH_METERS: Final[float] = 4.5       # Average car length (metres)
MIN_GAP_METERS: Final[float] = 2.0          # Minimum bumper-to-bumper gap when stopped
FOLLOW_TIME_HEADWAY: Final[float] = 1.5     # Seconds gap to maintain (time headway)
MAX_CATCHUP_SPEED_DELTA: Final[float] = 2.0  # Max m/s faster than the lead car
LOOK_AHEAD_FACTOR: Final[float] = 3.0       # Multiplier on desired gap — beyond this, no influence
COMFORTABLE_DECELERATION: Final[float] = 3.0  # m/s^2 — comfortable braking
EMERGENCY_DECELERATION: Final[float] = 6.0  # m/s^2 — hard braking to avoid collision
COMFORTABLE_ACCELERATION: Final[float] = 2.0  # m/s^2 — normal acceleration

# ── Road rendering ────────────
ROAD_COLOR: Final[str] = "#2c3e50"  # dark grey
ROAD_BLOCKED_COLOR: Final[str] = "#e74c3c"  # red
ROAD_HIGHLIGHT_COLOR: Final[str] = "#f39c12"  # orange (hover)
LANE_WIDTH_PX: Final[int] = 4
ROAD_MIN_WIDTH_PX: Final[int] = 6
MIN_ROAD_LENGTH_PX: Final[float] = 10.0

# ── Traffic light defaults ────
TRAFFIC_LIGHT_GREEN_SECONDS: Final[float] = 5.0
TRAFFIC_LIGHT_YELLOW_SECONDS: Final[float] = 1.0
TRAFFIC_LIGHT_RED_SECONDS: Final[float] = 5.0
TRAFFIC_LIGHT_RADIUS_PX: Final[int] = 4
TRAFFIC_LIGHT_GREEN_COLOR: Final[str] = "#2ecc71"
TRAFFIC_LIGHT_YELLOW_COLOR: Final[str] = "#f1c40f"
TRAFFIC_LIGHT_RED_COLOR: Final[str] = "#e74c3c"

# ── UI ────────────────────────
WINDOW_WIDTH: Final[int] = 1280
WINDOW_HEIGHT: Final[int] = 800
UI_PANEL_WIDTH: Final[int] = 280
UI_BG_COLOR: Final[str] = "#1a1a2e"
UI_TEXT_COLOR: Final[str] = "#ecf0f1"
UI_ACCENT_COLOR: Final[str] = "#3498db"
UI_BUTTON_COLOR: Final[str] = "#2c3e50"
UI_BUTTON_HOVER_COLOR: Final[str] = "#34495e"
UI_SLIDER_TRACK_COLOR: Final[str] = "#7f8c8d"
UI_SLIDER_HANDLE_COLOR: Final[str] = "#3498db"
UI_FONT_SIZE: Final[int] = 16
UI_FONT_SIZE_SMALL: Final[int] = 12
UI_FONT_SIZE_LARGE: Final[int] = 20

# ── Map defaults ──────────────
DEFAULT_CITY: Final[str] = "Manhattan, New York, USA"
MAX_DOWNLOAD_RETRIES: Final[int] = 3
OSM_CACHE_DIR: Final[str] = "data"
DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 30

# ── Pathfinding ───────────────
AVAILABLE_ALGORITHMS: Final[list[str]] = [
    "dijkstra",
    "a_star",
    "bfs",
    "yen_k_shortest",
    "selfish",
]
DEFAULT_ALGORITHM: Final[str] = "dijkstra"
YEN_K_DEFAULT: Final[int] = 3  # number of alternative routes for Yen's algorithm

# ── BPR congestion function (selfish algorithm) ──
# travel_time = free_flow_time * (1 + 0.15 * (v / capacity)^4)
BPR_ALPHA: Final[float] = 0.15       # BPR parameter α
BPR_BETA: Final[float] = 4.0         # BPR parameter β (exponent)
CAPACITY_PER_LANE: Final[int] = 8    # "Cars" per lane before congestion sets in
SELFISH_REROUTE_INTERVAL: Final[float] = 5.0  # seconds between selfish reroute cycles

# ── Colors (Pygame tuples) ────
# These are computed lazily; import the COLORS dict after pygame init.
# Tuples are (R, G, B) 0-255.
_COLOR_HEX_MAP: dict[str, tuple[int, int, int]] = {
    "#3498db": (52, 152, 219),
    "#e74c3c": (231, 76, 60),
    "#2c3e50": (44, 62, 80),
    "#f39c12": (243, 156, 18),
    "#2ecc71": (46, 204, 113),
    "#f1c40f": (241, 196, 15),
    "#1a1a2e": (26, 26, 46),
    "#ecf0f1": (236, 240, 241),
    "#7f8c8d": (127, 140, 141),
    "#34495e": (52, 73, 94),
    "#ffffff": (255, 255, 255),
    "#000000": (0, 0, 0),
}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex colour string to an (R, G, B) tuple."""
    return _COLOR_HEX_MAP[hex_color]