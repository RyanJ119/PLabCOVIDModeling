"""Open this file in Spyder and press F5 to run the HPV PDE model.

Results remain available in Spyder's Variable Explorer. All parameter values
except Census age/birth inputs are illustrative; consult README.md before interpretation.
"""
from pathlib import Path
import sys
import numpy as np
import json
from dataclasses import asdict

# Locate companion code and data beside this file, independent of Spyder's working directory.
PROJECT_FOLDER = Path(__file__).resolve().parent
if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))
from hpv_pde import HPVModel, Parameters
from census_demography import birth_inflow

# %% 1. Run settings
# Time is in years. SAVE_EVERY is the output interval; the solver takes smaller steps.
YEARS = 40.0  # simulation duration, years

SAVE_EVERY = 1.0  # years between saved snapshots
MAKE_PLOTS = True  # display figures and save PNG images
TRACK_INITIAL_POPULATION = True  # track initially present people separately from later births
RUN_SCENARIOS = True  # set False to skip extra comparison simulations

# Scenarios change only vaccination rates. Existing vaccinated people remain
# in every scenario, including "No new vaccination".
SCENARIO_POLICIES = {
    'No new vaccination': (0.0,)*5,
    'Moderate uptake': (0.05,)*5,
    'High uptake': (0.5,)*5,
}
OUTPUT_FOLDER = PROJECT_FOLDER / 'spyder_results'

# %% 2. Demographic inputs
# Local Census 2025 ages, both sexes combined. The source covers ages 0--99.
INITIAL_AGE_DATA_FILE = PROJECT_FOLDER / 'census_data' / 'us_2025_age_counts.csv'

# Raw people/year: Census NC-EST2025-COMPN!I7, July 2024--July 2025.
ANNUAL_BIRTHS = 3620461
INITIAL_POPULATION = 1.0  # 1 means population is measured relative to its initial size
MAX_AGE = 100.0  # years; local Census ages cover [0,100)
N_AGE = 100  # number of age cells
# Convert raw births to model units: initial population * births / represented Census count.
# For initial population 1 and maximum age 100, this is about 0.0105965/year.
CENSUS_NEWBORN_RATE = birth_inflow(INITIAL_AGE_DATA_FILE, ANNUAL_BIRTHS,
                                  INITIAL_POPULATION, MAX_AGE, N_AGE)

# %% 3. Model parameters: edit values here for Spyder runs
# All model fields are listed here; hpv_pde.py supplies defaults for standalone use.
# Six-entry tuples use: one dose <12; one dose >=12; two doses <12;
# two doses 12--17; two doses 18--26; unvaccinated.
# The five vaccination rates use the same order, excluding unvaccinated.
parameters = Parameters(

    max_age=MAX_AGE,  # maximum modeled age, years
    n_age=N_AGE,  # age-cell count
    n_x=100,  # disease-cell count on [0,1]
    cfl=0.8,  # time-step safety factor in (0,1]

    # Infection pressure beta(a) * I: beta(a) is a bell-shaped age profile.
    beta_peak=0.22,  # disease coordinate / (population unit * year)
    beta_age_center=27.0,  # age of peak infection coefficient, years
    beta_age_width=20.0,  # width of the age profile, years

    # Restorative drift toward health, in disease-coordinate units/year.
    # The last value applies to unvaccinated people; these values are not percentages.
    clearance=(0.15, 0.03, 0.1, 0.08, 0.04, 0.001),
    clearance_age_scale=80.0,  # age scale reducing clearance, years

    # Treatment subtracts gamma * effort from the disease-progression velocity.
    gamma=0.05,  # restorative drift per unit treatment effort
    treatment_effort=0.0,  # nonnegative treatment effort

    # Annual transfer rates, not exact annual percentages. Overlapping eligible
    # pathways add: a total rate h gives probability 1-exp(-h) over one year
    # if eligibility is unchanged and there are no other exits. Two-dose entry
    # is currently a direct status transfer, not two sequential appointments.
    vaccination_portion_per_year=(0.05, 0.05, 0.05, 0.05, 0.05),

    # Initial joint distribution = group fractions * age profile * disease profiles.
    # The age CSV overrides initial_age_scale. 
    initial_population=INITIAL_POPULATION,  # population units
    initial_group_fractions=(0.02, 0.02, 0.02, 0.02, 0.02, 0.90),  # six initial proportions; must sum to 1
    initial_age_data_file=str(INITIAL_AGE_DATA_FILE),  # local empirical age counts
    initial_age_scale=35.0,  # years; used only if age data file is None
    initial_x_means=(0.08, 0.10, 0.12, 0.14, 0.16, 0.25),  # six Gaussian centers, before truncation to [0,1]
    initial_x_width=0.08,  # Gaussian width in disease-coordinate units

    # Births enter unvaccinated, near disease state zero. Natural mortality
    # is mu(age) = mortality_at_birth * exp(mortality_growth * age).
    newborn_rate=CENSUS_NEWBORN_RATE,  # constant birth influx, model population units/year
    mortality_at_birth=0.0001,  # mortality hazard at age zero, per year
    mortality_growth=0.08,  # exponential growth per year of age
)

# %% 4. Run the model
# The optional tag follows people present at time zero under the same dynamics.
model = HPVModel(parameters)

# density[time_index, group_index, age_index, disease_index] is a DENSITY.
# Multiply by age-cell width * disease-cell width to get population in a cell.
# Example: density[-1, 5, :, :] selects unvaccinated people at the final time.
time, density, summary = model.simulate(years=YEARS, save_every=SAVE_EVERY,
                                        track_initial=TRACK_INITIAL_POPULATION)
initial_cohort_age_density = model.initial_cohort_age_density
scenario_results = []
# Reuse the current solution if a scenario has the same vaccination rates.
if RUN_SCENARIOS:
    from scenario_comparison import run_scenarios
    scenario_results = run_scenarios(parameters, YEARS, SAVE_EVERY,
                                     SCENARIO_POLICIES, time, summary)
age = model.age
x = model.x

# Each summary slice is one value per saved time. Full column meanings are
# in model_description.tex; burden is severity-weighted mass, not prevalence.
population = summary[:, 1]
burden = summary[:, 2]
cumulative_cancer = summary[:, 3]
cumulative_natural_deaths = summary[:, 18]
mortality_by_age = model.mortality

# %% 5. Save results
# NPZ files contain numerical arrays; JSON files record settings and sources.
# Re-running overwrites files with these names. Edit inputs above, not output JSON.
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
cohort_data = ({'initial_cohort_age_density': initial_cohort_age_density}
               if initial_cohort_age_density is not None else {})
np.savez_compressed(OUTPUT_FOLDER / 'solution.npz', time=time, age=age,
                    x=x, density=density, summary=summary, mortality_by_age=mortality_by_age, **cohort_data)

(OUTPUT_FOLDER / 'demographic_sources.json').write_text(
    json.dumps(dict(json.loads((PROJECT_FOLDER / 'census_data' / 'sources.json').read_text()),
                    applied_annual_births=ANNUAL_BIRTHS,
                    applied_newborn_rate=parameters.newborn_rate,
                    applied_age_file=parameters.initial_age_data_file), indent=2) + '\n')

(OUTPUT_FOLDER / 'parameters.json').write_text(
    json.dumps(asdict(parameters), indent=2) + '\n')
if scenario_results:
    np.savez_compressed(OUTPUT_FOLDER / 'scenario_comparison.npz', time=time,
                        labels=np.array([r['label'] for r in scenario_results]),
                        policies=np.array([r['policy'] for r in scenario_results]),
                        summaries=np.array([r['summary'] for r in scenario_results]))
    (OUTPUT_FOLDER / 'scenario_policies.json').write_text(
        json.dumps([dict(label=r['label'], policy=r['policy']) for r in scenario_results],
                   indent=2) + '\n')
print(f'Finished {YEARS:g} years. Results saved to {OUTPUT_FOLDER}')
print('Available variables: time, density, summary, age, x, population, burden, cumulative_cancer, cumulative_natural_deaths, mortality_by_age')

# %% 6. Display and save plots
# PNG images go in spyder_results/plots. Figures 04 and 09 are retired.
if MAKE_PLOTS:
    from plot_hpv import plot_results
    figures = plot_results(time, age, x, density, summary,
                           OUTPUT_FOLDER / 'plots', mortality=mortality_by_age,
                           scenarios=scenario_results,
                           initial_cohort_age_density=initial_cohort_age_density)
