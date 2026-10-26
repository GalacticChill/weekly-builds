"""Data-generating processes with a *known* causal effect.

The only honest way to test a causal method is to build a world where you already
know the right answer, then see whether the method recovers it. Each function here
simulates one canonical causal structure and returns the data plus the true effect
of the treatment T on the outcome Y. The three structures differ only in the role of
one extra variable -- and that role is everything:

- **Confounder** (C -> T, C -> Y): a common *cause* of treatment and outcome. It
  creates a spurious association you must remove by adjusting for it.
- **Collider** (T -> Z <- Y): a common *effect* of treatment and outcome. Adjusting
  for it *creates* a spurious association out of nothing.
- **Mediator** (T -> M -> Y): a step on the causal path. Adjusting for it removes
  part of the real effect, answering a different question than you probably meant.

Treatments are continuous so the causal effect is simply a regression coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    data: pd.DataFrame
    true_effect: float          # the real causal effect of T on Y (total effect)
    extra: str                  # name of the third variable (C / Z / M)
    adjusting_is_right: bool    # should you adjust for `extra` to get the true effect?
    note: str


def confounded(n: int = 4000, effect: float = 0.5, seed: int = 0) -> Scenario:
    """C -> T, C -> Y, T -> Y. Naive T~Y is biased; adjust for C to fix it."""
    rng = np.random.default_rng(seed)
    c = rng.normal(0, 1, n)
    t = 0.8 * c + rng.normal(0, 1, n)
    y = effect * t + 1.0 * c + rng.normal(0, 1, n)
    return Scenario(
        data=pd.DataFrame({"T": t, "Y": y, "C": c}),
        true_effect=effect, extra="C", adjusting_is_right=True,
        note="C is a common cause; adjusting removes the backdoor path.",
    )


def collider(n: int = 4000, effect: float = 0.0, seed: int = 0) -> Scenario:
    """T -> Z <- Y, with the true T->Y effect set by `effect` (default 0).

    Z is a common effect of T and Y. Adjusting for Z opens a spurious path and
    biases the estimate; the naive estimate is the correct one.
    """
    rng = np.random.default_rng(seed)
    t = rng.normal(0, 1, n)
    y = effect * t + rng.normal(0, 1, n)
    z = 0.8 * t + 0.8 * y + rng.normal(0, 1, n)
    return Scenario(
        data=pd.DataFrame({"T": t, "Y": y, "Z": z}),
        true_effect=effect, extra="Z", adjusting_is_right=False,
        note="Z is a common effect; adjusting manufactures a spurious association.",
    )


def mediated(n: int = 4000, direct: float = 0.2, a: float = 0.7, b: float = 0.6,
             seed: int = 0) -> Scenario:
    """T -> M -> Y plus a direct T -> Y. Total effect = direct + a*b.

    Adjusting for the mediator M removes the indirect path, leaving only the direct
    effect -- a correct answer to a *different* question (direct vs total effect).
    """
    rng = np.random.default_rng(seed)
    t = rng.normal(0, 1, n)
    m = a * t + rng.normal(0, 1, n)
    y = direct * t + b * m + rng.normal(0, 1, n)
    total = direct + a * b
    return Scenario(
        data=pd.DataFrame({"T": t, "Y": y, "M": m}),
        true_effect=total, extra="M", adjusting_is_right=False,
        note=f"Adjusting for M yields the direct effect ({direct:.2f}), not the "
             f"total ({total:.2f}).",
    )


SCENARIOS = {
    "confounder": confounded,
    "collider": collider,
    "mediator": mediated,
}
