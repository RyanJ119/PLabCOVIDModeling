"""Offline Census age-profile loading and population-preserving grid mapping."""
from pathlib import Path
import csv
import numpy as np

def age_counts_on_grid(filename, max_age, n_age):
    """Return people per model age cell, assuming uniform density within source bands.

    The packaged data cover ages [0,100). Refuse extrapolation beyond available
    ages: the original 100+ category cannot be split into single years reliably.
    """
    # Read local columns age_start, age_end, population. Counts are people, not fractions.
    with Path(filename).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Age data file is empty')
    lo = np.array([float(r['age_start']) for r in rows])
    hi = np.array([float(r['age_end']) for r in rows])
    counts = np.array([float(r['population']) for r in rows])
    # Require complete, contiguous age bands; do not silently fill missing ages.
    if (not np.all(np.isfinite(np.concatenate((lo, hi, counts))))
            or np.any(hi <= lo) or np.any(counts < 0) or lo[0] != 0
            or not np.allclose(lo[1:], hi[:-1], rtol=0, atol=1e-10)):
        raise ValueError('Age bands must be ordered, contiguous, finite, and nonnegative')
    if not 0 < max_age <= hi[-1] or n_age < 2:
        raise ValueError(f'Age data cover [0,{hi[-1]:g}); max_age must lie in that range')
    # For each model cell, measure how much it overlaps every source age band.
    # This preserves counts even when model cells do not match one-year data bins.
    edges = np.linspace(0, max_age, n_age+1)
    overlap = np.maximum(0, np.minimum(edges[1:, None], hi[None, :])
                         - np.maximum(edges[:-1, None], lo[None, :]))
    # Assume uniform density inside each source band: count / band width.
    # @ is matrix multiplication, summing all overlapping source contributions.
    cell_counts = overlap @ (counts/(hi-lo))
    if cell_counts.sum() <= 0:
        raise ValueError('Age data contain no population in the model domain')
    return cell_counts

def birth_inflow(filename, annual_births, initial_population, max_age, n_age):
    """Scale observed annual births into model population units (constant inflow)."""
    if not np.isfinite(annual_births) or annual_births < 0:
        raise ValueError('annual_births must be finite and nonnegative')
    # Scale births consistently with the population represented below max_age.
    # The result is model population units/year, held constant during the simulation.
    represented_people = age_counts_on_grid(filename, max_age, n_age).sum()
    return initial_population*annual_births/represented_people
