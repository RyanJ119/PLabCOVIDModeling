"""Age/disease finite-volume solver for root.tex; see README.md before use.

All defaults are illustrative, uncalibrated placeholders. Requires only NumPy.
Run: python3 hpv_pde.py --years 10 --output results
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import numpy as np

# This fixed ordering is shared by parameter tuples, state arrays, and saved outputs.
# Python indices 0--4 are vaccinated groups; index 5 is unvaccinated.
GROUPS = ('one_dose_before_12', 'one_dose_after_12', 'two_doses_before_12',
          'two_doses_12_to_18', 'two_doses_18_to_26', 'unvaccinated')

# Default settings for standalone use. Spyder users should edit run_spyder.py.
@dataclass
class Parameters:

    # Domain and grid: more cells improve resolution but require more computation.
    max_age: float = 100.0
    n_age: int = 100
    n_x: int = 100

    # Infection coefficient beta(a): Gaussian peak, center age, and age width.
    beta_peak: float = 0.12
    beta_age_center: float = 27.0
    beta_age_width: float = 15.0

    # Six restorative drifts in GROUPS order; reduced with age by clearance_age_scale.
    clearance: tuple = (0.10, 0.08, 0.06, 0.04, 0.025, 0.01)
    clearance_age_scale: float = 80.0

    gamma: float = 0.05
    treatment_effort: float = 0.0

    # Five annual vaccination hazards, restricted to eligible ages.
    # Despite the name, these are rates, not annual fractions or vaccine efficacies.
    vaccination_portion_per_year: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)

    # Population scale, initial group composition, and age/disease profiles.
    initial_population: float = 1.0
    initial_group_fractions: tuple = (0.02, 0.02, 0.02, 0.02, 0.02, 0.90)
    initial_age_scale: float = 35.0
    initial_age_data_file: str | None = None
    initial_x_means: tuple = (0.08, 0.10, 0.12, 0.14, 0.16, 0.25)
    initial_x_width: float = 0.08
    # Demographic inputs: constant birth influx and exponential natural mortality.
    newborn_rate: float = 0.01

    mortality_at_birth: float = 0.0001
    mortality_growth: float = 0.08
    cfl: float = 0.8

class HPVModel:
    def __init__(self, params=None):
        self.par = params or Parameters()
        p = self.par

        # Check inputs before building arrays; invalid settings raise descriptive errors.
        for key in ('max_age', 'beta_age_width', 'clearance_age_scale',
                    'initial_age_scale', 'initial_x_width', 'initial_population'):
            if not np.isfinite(getattr(p, key)) or getattr(p, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('n_age', 'n_x'):
            if not isinstance(getattr(p, key), int) or getattr(p, key) < 2:
                raise ValueError(f'{key} must be an integer >= 2')
        for key, size in (('clearance', 6), ('vaccination_portion_per_year', 5),
                          ('initial_group_fractions', 6), ('initial_x_means', 6)):
            v = np.asarray(getattr(p, key), dtype=float)
            if v.shape != (size,) or not np.all(np.isfinite(v)) or np.any(v < 0):
                raise ValueError(f'{key} needs {size} finite nonnegative values')
        if not np.isclose(sum(p.initial_group_fractions), 1):
            raise ValueError('Initial group fractions must sum to one')
        if np.any(np.asarray(p.initial_x_means) > 1):
            raise ValueError('Initial disease means must lie in [0, 1]')
        for key in ('beta_peak', 'gamma', 'treatment_effort', 'newborn_rate',
                    'mortality_at_birth', 'mortality_growth'):
            if not np.isfinite(getattr(p, key)) or getattr(p, key) < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if not np.isfinite(p.beta_age_center):
            raise ValueError('beta_age_center must be finite')
        if not 0 < p.cfl <= 1:
            raise ValueError('cfl must be in (0, 1]')

        # A single state has axes [group, age, disease]. da and dx are cell widths.
        # Coordinates are cell centers: x starts at dx/2, not exactly zero.
        self.da, self.dx = p.max_age / p.n_age, 1 / p.n_x
        self.age = (np.arange(p.n_age) + 0.5) * self.da
        self.x = (np.arange(p.n_x) + 0.5) * self.dx

        # Eligibility is checked only when transferring out of unvaccinated.
        # Aging does not reassign existing vaccination-history groups.
        # Age bands use cell centers; align grid faces with 12, 18, and 27 if possible.
        self.vaccination_eligibility = np.stack((
            self.age < 12,
            self.age >= 12,
            self.age < 12,
            (self.age >= 12) & (self.age < 18),
            (self.age >= 18) & (self.age < 27),
        ))
        self.beta = p.beta_peak * np.exp(-0.5 * ((self.age-p.beta_age_center)/p.beta_age_width)**2)

        # Evaluate mu(a) = mu0 * exp(g*a); check in log space to avoid overflow.
        if p.mortality_at_birth == 0:
            self.mortality = np.zeros_like(self.age)
        else:
            log_mortality = np.log(p.mortality_at_birth) + p.mortality_growth*self.age
            if np.any(log_mortality > np.log(np.finfo(float).max)):
                raise ValueError('Mortality parameters overflow on the age grid')
            self.mortality = np.exp(log_mortality)

        # None inserts an axis for broadcasting: (6,1) baseline rates combine
        # with ages to produce one clearance value per group and age, shape (6,n_age).
        self.nu = np.asarray(p.clearance)[:, None] / (1+self.age[None, :]/p.clearance_age_scale)

    def initial_state(self):
        """Build initial density [group, age, disease] with total mass initial_population."""
        p = self.par

        # Normalize the age profile so sum(age_density * da) = 1.
        # Census data override the optional exponential profile.
        if p.initial_age_data_file is None:
            age_density = np.exp(-self.age / p.initial_age_scale)
            age_density /= age_density.sum() * self.da
        else:

            from census_demography import age_counts_on_grid
            cell_counts = age_counts_on_grid(p.initial_age_data_file, p.max_age, p.n_age)
            age_density = cell_counts/(cell_counts.sum()*self.da)
        # Normalize each truncated Gaussian so sum(disease[k] * dx) = 1.
        # Its center is an input, not necessarily its actual mean after truncation.
        disease = np.exp(-0.5*((self.x[None, :]-np.asarray(p.initial_x_means)[:, None])/p.initial_x_width)**2)
        disease /= disease.sum(axis=1)[:, None] * self.dx

        # Combine profiles to give initial state shape (6,n_age,n_x).
        # Initial vaccination fractions are assumed independent of age; this can
        # assign older-vaccination histories to children. Replace that assumption
        # with an age-specific joint distribution when such data become available.
        return (p.initial_population * np.asarray(p.initial_group_fractions)[:, None, None]
                * age_density[None, :, None] * disease[:, None, :])

    def burden(self, state):

        # Integrate x*p over age, disease, and groups. This is raw burden I, not I/N.
        # The first disease cell contributes a small burden because its center is dx/2.
        return float(np.sum(state * self.x[None, None, :]) * self.da * self.dx)

    def velocity(self, state):

        # Positive velocity moves toward cancer; negative velocity moves toward health.
        # All ages/groups contribute to the shared burden driving infection pressure.
        return self.beta[None, :] * self.burden(state) - self.nu - self.par.gamma*self.par.treatment_effort

    def step(self, state, max_dt, *, velocity=None, include_births=True):
        """Advance one stable step; return density, dt, cancer rates, and age-exit rate."""

        # Use full-population velocity for a tagged cohort so its transmission
        # environment is the same as everyone else's.
        v = self.velocity(state) if velocity is None else velocity
        hazards = np.asarray(self.par.vaccination_portion_per_year)

        # The combined CFL bound limits transport and source losses in one step.
        # This preserves nonnegative density; output spacing does not determine dt.
        loss_bound = 1/self.da + np.max(np.abs(v))/self.dx + hazards.sum() + self.mortality.max()
        dt = min(max_dt, self.par.cfl / loss_bound)

        # Disease fluxes live on cell faces: n_x cells require n_x+1 faces.
        # The zero-initialized first face retains people at the healthy boundary.
        fx = np.zeros((6, self.par.n_age, self.par.n_x+1))

        # Upwinding uses the density on the side from which people are moving.
        # Positive velocity selects the left cell; negative velocity selects the right.
        fx[:, :, 1:-1] = np.maximum(v, 0)[:, :, None]*state[:, :, :-1] + np.minimum(v, 0)[:, :, None]*state[:, :, 1:]

        # At x=1 only outward flow is allowed. This records cancer onset, not death.
        fx[:, :, -1] = np.maximum(v, 0)*state[:, :, -1]

        # Age velocity is 1 year/year. The final age face is an exit at max_age,
        # recorded separately from natural mortality.
        fa = np.zeros((6, self.par.n_age+1, self.par.n_x))
        fa[:, 1:, :] = state

        # Only newborns enter at age zero, in group 5 and the first disease cell.
        # Dividing by dx makes the total age-zero flux equal newborn_rate.
        # Tagged initial cohorts have no birth influx.
        fa[5, 0, 0] = self.par.newborn_rate / self.dx if include_births else 0.0

        # Per-target rate * eligibility * available unvaccinated density.
        # Transfers change group only, preserving current age and disease state.
        transfer = (hazards[:, None, None] * self.vaccination_eligibility[:, :, None]
                    * state[5][None, :, :])

        # Recipient gains exactly equal unvaccinated losses: vaccination conserves mass.
        source = np.zeros_like(state)
        source[:5] = transfer
        source[5] = -transfer.sum(axis=0)

        # PDE: minus net outward disease/age flux + vaccination - natural deaths.
        # Dividing flux differences by cell widths converts them to density-change rates.
        rhs = -(fx[:, :, 1:]-fx[:, :, :-1])/self.dx - (fa[:, 1:, :]-fa[:, :-1, :])/self.da + source - self.mortality[None, :, None]*state

        # Forward Euler update: density at next time = density now + dt * change rate.
        updated = state + dt*rhs
        if updated.min() < -1e-12:
            raise RuntimeError('Negative density: check the discretization and controls')

        # Integrate boundary fluxes to obtain exit RATES. simulate() accumulates
        # rate * dt to produce cumulative exits.
        cancer_by_group = fx[:, :, -1].sum(axis=1) * self.da
        age_outflow = float(fa[:, -1, :].sum() * self.dx)
        return updated, dt, cancer_by_group, age_outflow

    def simulate(self, years=10.0, save_every=1.0, initial=None, track_initial=False):
        """Return saved times, density snapshots, and summary rows.

        initial: optional density array [group, age, disease].
        track_initial: additionally store the initial cohort's age marginal in
        self.initial_cohort_age_density [saved time, age]; return values stay the same.
        """
        if not np.isfinite(years) or years <= 0 or not np.isfinite(save_every) or save_every <= 0:
            raise ValueError('years and save_every must be finite and positive')

        # Accept optional measured initial density without modifying the caller's array.
        state = self.initial_state() if initial is None else np.array(initial, dtype=float, copy=True)
        if state.shape != (6, self.par.n_age, self.par.n_x) or not np.all(np.isfinite(state)) or np.any(state < 0):
            raise ValueError('initial must be finite, nonnegative, shape (6, n_age, n_x)')

        times, snapshots, rows = [0.0], [state.copy()], []

        # The tag marks everyone present initially, and follows their survivors.
        # Only its age marginal is stored; it has no effect on the total solution.
        tagged = state.copy() if track_initial else None
        tagged_ages = [tagged.sum(axis=(0, 2))*self.dx] if track_initial else []
        self.initial_cohort_age_density = None
        cancer = np.zeros(6)
        age_loss = 0.0
        deaths = 0.0
        t = 0.0

        # Summary columns (zero-based): 0 time, 1 population, 2 burden, 3 cancer,
        # 4 age exits, 5 births, 6--11 group populations, 12--17 group cancers, 18 deaths.
        def record():
            mass = state.sum(axis=(1, 2))*self.da*self.dx
            rows.append([t, mass.sum(), self.burden(state), cancer.sum(), age_loss,
                         self.par.newborn_rate*t, *mass, *cancer, deaths])
        record()
        target = min(save_every, years)
        while t < years:

            # Use pre-step densities and fluxes consistently for both the state update
            # and cumulative exits; this preserves the population accounting identity.
            death_rate = float(np.sum(state*self.mortality[None, :, None])*self.da*self.dx)
            shared_velocity = self.velocity(state)
            state, dt, incidence, old_age = self.step(state, target-t, velocity=shared_velocity)
            # Advance the initial-cohort tag with identical velocity and dt, but no births.
            if track_initial:
                tagged, tagged_dt, _, _ = self.step(tagged, dt, velocity=shared_velocity,
                                                    include_births=False)
                if not np.isclose(tagged_dt, dt, rtol=0, atol=1e-14):
                    raise RuntimeError('Cohort and total time steps must agree')

            deaths += dt*death_rate
            cancer += dt*incidence
            age_loss += dt*old_age
            t += dt
            # Store a copy at each requested output time; internal steps are usually smaller.
            if t >= target-1e-12:
                t = target
                times.append(t)

                snapshots.append(state.copy())
                if track_initial:
                    tagged_ages.append(tagged.sum(axis=(0, 2))*self.dx)
                record()
                if target == years:
                    break
                target = min(target+save_every, years)
        if track_initial:
            self.initial_cohort_age_density = np.array(tagged_ages)
        return np.array(times), np.array(snapshots), np.array(rows)

def main():

    # Terminal interface. Importing HPVModel from Spyder does not run this function.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years', type=float, default=10.0)
    parser.add_argument('--save-every', type=float, default=1.0)
    parser.add_argument('--config', type=Path, help='JSON overrides for Parameters')
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    overrides = json.loads(args.config.read_text()) if args.config else {}
    model = HPVModel(Parameters(**overrides))
    times, states, rows = model.simulate(args.years, args.save_every)
    args.output.mkdir(parents=True, exist_ok=True)

    # Save the full solution as named arrays and a smaller tabular summary as CSV.
    np.savez_compressed(args.output/'solution.npz', time=times, age=model.age,
                        x=model.x, density=states, summary=rows, mortality_by_age=model.mortality, groups=np.array(GROUPS))
    columns = ['time', 'population', 'burden', 'cumulative_cancer', 'cumulative_age_outflow', 'cumulative_births']
    columns += ['population_'+g for g in GROUPS] + ['cumulative_cancer_'+g for g in GROUPS]
    columns += ['cumulative_natural_deaths']
    with (args.output/'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    from dataclasses import asdict
    (args.output/'parameters.json').write_text(json.dumps(asdict(model.par), indent=2)+'\n')
    print(f'Saved {len(times)} snapshots to {args.output.resolve()}')
    print(f'Final population={rows[-1,1]:.6g}; burden={rows[-1,2]:.6g}; cumulative cancer={rows[-1,3]:.6g}')

if __name__ == '__main__':
    main()
