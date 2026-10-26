"""causal_traps: correlation, causation, and the control-for-everything trap.

Simulate three canonical causal structures with known effects -- confounder,
collider, mediator -- and show that adjusting for the third variable fixes the
estimate in one case, ruins it in another, and silently changes the question in the
third.
"""

from .dgp import SCENARIOS, Scenario, collider, confounded, mediated
from .estimate import Fit, effect, effect_fit, ols
from .experiment import ScenarioResult, run_all, run_scenario

__all__ = [
    "confounded",
    "collider",
    "mediated",
    "Scenario",
    "SCENARIOS",
    "ols",
    "effect",
    "effect_fit",
    "Fit",
    "run_scenario",
    "run_all",
    "ScenarioResult",
]
