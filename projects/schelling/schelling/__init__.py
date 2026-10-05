"""schelling: how mild individual preferences cascade into a segregated city.

An agent-based simulation of Thomas Schelling's segregation model, with a tolerance
sweep that locates the tipping point and measures of how divided the city becomes.
"""

from .experiment import SweepResult, sweep
from .model import (
    A,
    B,
    EMPTY,
    RunResult,
    discontent_mask,
    happy_fraction,
    make_grid,
    run,
    same_type_share,
    segregation,
)

__all__ = [
    "EMPTY",
    "A",
    "B",
    "make_grid",
    "same_type_share",
    "discontent_mask",
    "segregation",
    "happy_fraction",
    "run",
    "RunResult",
    "sweep",
    "SweepResult",
]
