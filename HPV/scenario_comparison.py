"""Controlled comparisons: vary only vaccination uptake, keeping all else fixed."""
from dataclasses import replace
import numpy as np
from hpv_pde import HPVModel

def run_scenarios(parameters, years, save_every, policies, time, current_summary):
    """Return labeled results; reuse the current run when a policy matches it."""
    results = []
    # Each policy supplies five annual transfer rates; other parameters stay fixed.
    current_policy = tuple(parameters.vaccination_portion_per_year)
    for label, policy in policies.items():
        if len(policy) != 5:
            raise ValueError('Every scenario needs five vaccination rates')
        # Reuse a matching baseline rather than repeating the same simulation.
        if tuple(policy) == current_policy:
            summary = current_summary
            scenario_time = time
            label += ' (current)'
        else:
            # replace() creates a new parameter object, leaving the user's settings unchanged.
            # Each scenario recalculates its own transmission pressure over time.
            model = HPVModel(replace(parameters, vaccination_portion_per_year=tuple(policy)))
            scenario_time, _, summary = model.simulate(years, save_every)
        # Ensure summaries share the time grid used by the comparison figure.
        np.testing.assert_allclose(scenario_time, time, rtol=0, atol=1e-12)
        results.append(dict(label=label, policy=list(policy), summary=summary))
    # Include current settings separately if they match none of the named scenarios.
    if current_policy not in [tuple(p) for p in policies.values()]:
        results.append(dict(label='Current settings', policy=list(current_policy),
                            summary=current_summary))
    return results
