"""flight_network: the world air-route network, robust yet fragile.

Build the airport graph from OpenFlights, measure its hubs and small-world path
length, and show that random failures barely dent it while targeted hub removal
shatters it.
"""

from .data import build_graph, build_snapshot, load, save
from .graph import (
    average_path_length,
    degree_distribution,
    degrees,
    giant_component_fraction,
)
from .robustness import (
    AttackCurve,
    attack,
    collapse_point,
    random_attack,
    targeted_attack,
)

__all__ = [
    "build_snapshot",
    "save",
    "load",
    "build_graph",
    "degrees",
    "giant_component_fraction",
    "degree_distribution",
    "average_path_length",
    "attack",
    "AttackCurve",
    "random_attack",
    "targeted_attack",
    "collapse_point",
]
