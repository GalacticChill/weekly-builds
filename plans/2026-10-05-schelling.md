# Week of 2026-10-05 — Schelling segregation (agent-based simulation)

**Theme:** data/simulation — the portfolio's first agent-based model.
**Type:** new technique (ABM / emergent-behavior simulation); user delegated ("do
the next one"), so a BUILD in weekly-builds (no new public repo on an ambiguous ask).

## The question
How much personal preference does it take to segregate a city? Rebuild Schelling's
1971 model and measure the tipping point.

## Scope (shipped)
- `model.py` — grid of two agent types + empties; `same_type_share` (8-neighbour,
  vectorized via shifted padded sums, NaN for empty/isolated); `discontent_mask`
  (below-tolerance agents); `segregation` (mean same-type share); `run` (each round
  discontent agents relocate to random empty cells until content or round cap).
- `experiment.py` — `sweep` over tolerance values x trials; `tipping_tolerance`
  (lowest tolerance reaching segregation >= 0.70).
- `plots.py` — before/after city maps (discrete cmap), tipping curve (with trial
  band + baseline + divided-city line), convergence over rounds.
- CLI; no external data (pure simulation, fully reproducible by seed).
- 13 offline deterministic tests.

## Findings
- At **33% tolerance** (agents happy as a 67% minority), a random city (seg 0.50)
  settles to **0.76** in ~32 rounds.
- Tipping curve: flat near 0.50 below ~20% preference, sharp rise crossing the
  "divided city" line (0.70) at ~30%, peaking ~0.96 at 60%.
- HONEST wrinkle: at 80% tolerance segregation falls back to ~0.51 — agents can
  never be satisfied, so the city churns forever (hits round cap, never converges).
  Flagged, not hidden.

## The honest core
- Mechanism not map: segregation can arise from mild preference alone, with no
  bigotry/policy/income effects — a sufficiency proof, not a claim it's the only
  cause.
- The extreme-tolerance turn-down is a no-equilibrium artifact (perpetual churn),
  explicitly explained.
- Random-relocation rule is a simplification (changes speed, not the story).

## Why this matters for the portfolio story
Adds agent-based modeling / emergent-behavior simulation (new technique), socially
meaningful ("society" interest), visually striking (before/after grids), and carries
a genuinely counterintuitive, essay-ready result. Pure numpy + matplotlib, no data
dependency, no new deps.

## Deliberately deferred / next
- Animated GIF of the settling process; three+ types; nearest-acceptable relocation.
- GRADUATE word-prints / benford / forecastable / this to standalone repos (confirm
  first — new public repo).
- Another original capstone.

## Status
Built and shipped 2026-10-05. 13 tests pass; verified end-to-end (50x50 run +
tolerance sweep). Three charts.
