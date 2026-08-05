"""Traffic light system with signal cycles.

Each traffic light sits at a node (intersection) and cycles through
green → yellow → red phases. Cars approaching a red light stop and wait.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import config

log = logging.getLogger(__name__)


class SignalPhase(Enum):
    """The current phase of a traffic light signal."""

    GREEN = auto()
    YELLOW = auto()
    RED = auto()


@dataclass
class TrafficLight:
    """A traffic light at an intersection.

    Controls the flow of traffic through a node by cycling through
    green → yellow → red phases.

    Attributes:
        node_id: The OSM node ID where this light is located.
        position: (lon, lat) position of the light.
        phase: Current signal phase.
        timer: Seconds elapsed in the current phase.
        green_seconds: Duration of green phase.
        yellow_seconds: Duration of yellow phase.
        red_seconds: Duration of red phase.
        affected_edges: List of (u, v, key) edges that this light controls.
            If empty, all edges into the node are controlled.
    """

    node_id: int
    position: tuple[float, float]
    phase: SignalPhase = SignalPhase.RED
    timer: float = 0.0
    green_seconds: float = config.TRAFFIC_LIGHT_GREEN_SECONDS
    yellow_seconds: float = config.TRAFFIC_LIGHT_YELLOW_SECONDS
    red_seconds: float = config.TRAFFIC_LIGHT_RED_SECONDS
    affected_edges: list[tuple[int, int, int]] = field(default_factory=list)

    def update(self, dt: float) -> None:
        """Advance the traffic light cycle by *dt* seconds.

        Args:
            dt: Delta time in seconds.
        """
        self.timer += dt
        cycle_duration = self.green_seconds + self.yellow_seconds + self.red_seconds

        # Normalise timer to cycle duration
        if self.timer >= cycle_duration:
            self.timer -= cycle_duration

        # Determine phase based on timer position
        if self.timer < self.green_seconds:
            self.phase = SignalPhase.GREEN
        elif self.timer < self.green_seconds + self.yellow_seconds:
            self.phase = SignalPhase.YELLOW
        else:
            self.phase = SignalPhase.RED

    def is_green(self) -> bool:
        """Return True if the light is green."""
        return self.phase == SignalPhase.GREEN

    def is_red(self) -> bool:
        """Return True if the light is red."""
        return self.phase == SignalPhase.RED

    def is_yellow(self) -> bool:
        """Return True if the light is yellow."""
        return self.phase == SignalPhase.YELLOW

    def get_color(self) -> tuple[int, int, int]:
        """Return the (R, G, B) colour for the current phase."""
        if self.phase == SignalPhase.GREEN:
            return config.hex_to_rgb(config.TRAFFIC_LIGHT_GREEN_COLOR)
        elif self.phase == SignalPhase.YELLOW:
            return config.hex_to_rgb(config.TRAFFIC_LIGHT_YELLOW_COLOR)
        else:
            return config.hex_to_rgb(config.TRAFFIC_LIGHT_RED_COLOR)

    def __repr__(self) -> str:
        return (
            f"TrafficLight(node={self.node_id}, "
            f"phase={self.phase.name}, timer={self.timer:.1f}s)"
        )