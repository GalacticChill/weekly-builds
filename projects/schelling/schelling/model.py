"""Schelling's segregation model — how tiny preferences become stark divides.

Thomas Schelling's 1971 question: if every household has only a *mild* preference
about its neighbors — say, wanting at least a third of the neighbors to be the same
type, perfectly content as a two-thirds minority — what does the city look like once
everyone has settled? The famous, uncomfortable answer is that mild individual
preferences cascade into near-total segregation. No one wanted it; it emerged anyway.

The mechanics here are deliberately simple:

- A grid of cells, each empty or home to one of two types of agent.
- An agent is **content** if the share of its occupied neighbours (the 8 surrounding
  cells) that match its own type is at least `tolerance`. Despite the name, a *higher*
  tolerance value means a *fussier* agent — it wants more same-type neighbours.
- Each round, every discontent agent moves to a random empty cell. Repeat until
  everyone is content or we hit a round cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EMPTY, A, B = 0, 1, 2


def make_grid(size: int = 50, empty_frac: float = 0.1, share_a: float = 0.5,
              seed: int = 0) -> np.ndarray:
    """A size x size grid randomly seeded with two agent types and empty cells."""
    rng = np.random.default_rng(seed)
    n = size * size
    n_empty = int(round(n * empty_frac))
    n_agents = n - n_empty
    n_a = int(round(n_agents * share_a))
    cells = np.array([EMPTY] * n_empty + [A] * n_a + [B] * (n_agents - n_a))
    rng.shuffle(cells)
    return cells.reshape(size, size)


def _neighbor_counts(mask: np.ndarray) -> np.ndarray:
    """Count, for every cell, how many of its 8 neighbours are set in `mask`."""
    padded = np.pad(mask.astype(int), 1)
    h, w = mask.shape
    total = np.zeros((h, w), dtype=int)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            total += padded[1 + di: 1 + di + h, 1 + dj: 1 + dj + w]
    return total


def same_type_share(grid: np.ndarray) -> np.ndarray:
    """For each occupied cell, the fraction of occupied neighbours of its own type.

    Cells that are empty, or that have no occupied neighbours, are returned as NaN
    (they don't contribute to happiness or segregation measures).
    """
    count_a = _neighbor_counts(grid == A)
    count_b = _neighbor_counts(grid == B)
    occupied_neighbors = count_a + count_b

    same = np.where(grid == A, count_a, np.where(grid == B, count_b, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(occupied_neighbors > 0, same / occupied_neighbors, np.nan)
    share[grid == EMPTY] = np.nan
    return share


def discontent_mask(grid: np.ndarray, tolerance: float) -> np.ndarray:
    """Occupied agents whose same-type neighbour share is below `tolerance`.

    An agent with no occupied neighbours at all is counted as content (it has no
    reason to move).
    """
    share = same_type_share(grid)
    occupied = grid != EMPTY
    has_neighbors = ~np.isnan(share)
    unhappy = occupied & has_neighbors & (share < tolerance)
    return unhappy


def segregation(grid: np.ndarray) -> float:
    """Mean same-type neighbour share across occupied cells (the segregation index).

    ~0.5 for a well-mixed 50/50 grid; approaches 1.0 as like clusters with like.
    """
    share = same_type_share(grid)
    vals = share[~np.isnan(share)]
    return float(vals.mean()) if vals.size else float("nan")


def happy_fraction(grid: np.ndarray, tolerance: float) -> float:
    """Share of occupied agents who are content at this tolerance."""
    occupied = int((grid != EMPTY).sum())
    if occupied == 0:
        return float("nan")
    return 1.0 - int(discontent_mask(grid, tolerance).sum()) / occupied


@dataclass
class RunResult:
    """The before/after grids and per-round trajectory of one simulation."""

    initial: np.ndarray = field(repr=False)
    final: np.ndarray = field(repr=False)
    tolerance: float
    rounds: int
    converged: bool
    segregation_history: list[float]
    happy_history: list[float]

    @property
    def final_segregation(self) -> float:
        return self.segregation_history[-1]

    @property
    def initial_segregation(self) -> float:
        return self.segregation_history[0]


def run(grid: np.ndarray, tolerance: float, max_rounds: int = 100,
        seed: int = 0) -> RunResult:
    """Run Schelling dynamics until everyone is content or `max_rounds` is hit.

    Each round, every discontent agent relocates to a randomly chosen empty cell.
    """
    rng = np.random.default_rng(seed)
    grid = grid.copy()
    initial = grid.copy()

    seg_hist = [segregation(grid)]
    happy_hist = [happy_fraction(grid, tolerance)]
    converged = False
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        unhappy = discontent_mask(grid, tolerance)
        movers = list(zip(*np.where(unhappy)))
        if not movers:
            converged = True
            rounds -= 1
            break

        empties = list(zip(*np.where(grid == EMPTY)))
        rng.shuffle(movers)
        rng.shuffle(empties)

        for (r, c) in movers:
            if not empties:
                break
            er, ec = empties.pop()
            grid[er, ec] = grid[r, c]
            grid[r, c] = EMPTY
            empties.append((r, c))

        seg_hist.append(segregation(grid))
        happy_hist.append(happy_fraction(grid, tolerance))

    return RunResult(
        initial=initial, final=grid, tolerance=tolerance, rounds=rounds,
        converged=converged, segregation_history=seg_hist, happy_history=happy_hist,
    )
