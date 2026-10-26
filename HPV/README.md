# HPV age–disease PDE solver

Source: `/Users/ryanweightman/Desktop/HPV/root.tex`, section “A Partial Differential Equation model for HPV progression in a population,” equations `eq:a-r-d` and `eq:a-r-d-cont`. Files generated here are in the writable project folder `/Users/ryanweightman/Documents/ChatGPT/HPV`.

## Run

Python 3.10+ and NumPy are required (`python3 -m pip install -r requirements.txt` if needed).

```sh
python3 hpv_pde.py --years 10 --output results
python3 hpv_pde.py --config example_parameters.json --years 20 --save-every 0.5
python3 -m unittest -v test_hpv_pde.py
```

**Every supplied numerical value is an illustrative placeholder, not an estimate from the manuscript or clinical data.** The manuscript's numerical ODE parameters cannot be substituted directly into this PDE.

`summary.csv` contains population, burden, cumulative cancer outflow, age outflow, births, and group totals. `solution.npz` contains cell-center ages, disease coordinates, times, group names, and `density[time, group, age, disease]`. `parameters.json` records the parameter configuration used. Average cancer incidence over an output interval is the difference in cumulative cancer divided by the interval duration. Cancer is an absorbing exit; individuals who reach it are no longer in the tracked density.

```python
import numpy as np
from hpv_pde import HPVModel, Parameters
model = HPVModel(Parameters())
time, density, summary = model.simulate(years=10, save_every=1)
# Model density is per year of age per unit disease coordinate.
initial = model.initial_state()  # replace with measured cell-average densities
```

## Implemented equations and numerical choices

The solver implements

    ∂t p_k + ∂a p_k + ∂x(v_k p_k) = R_k
    v_k = beta(a) I(t) - nu_k(a) - gamma(a) u1
    I(t) = sum_k integral integral x p_k da dx

Because velocity is independent of x, the conservative x term equals the manuscript's `v_k ∂x p_k` in the interior. An explicit first-order finite-volume upwind method uses an adaptive combined CFL bound for age transport, disease transport, and vaccination loss. This solves the structured PDE on an age/disease grid, rather than the paper's reduced four-state ODE. It is first-order accurate and numerically diffusive; refine both grids and compare results before inference.

The manuscript is incomplete as a numerical specification. These closure assumptions are explicit:

- Disease domain is `[0,1]`, following the population and burden integrals, despite an earlier mention of `[0,infinity)`.
- Age is truncated to `[0,max_age]`. Aging has unit speed; people exit at `max_age`, recorded separately from cancer. The displayed manuscript PDE has no natural mortality; the requested extension below adds a natural mortality sink.
- Zero net flux at `x=0` retains people reaching health in the first disease cell. This approximates a boundary mass; the original transport equation alone does not specify that boundary behavior. Zero external inflow at `x=1`; positive velocity causes cancer outflow, negative velocity causes no external inflow.
- Age-zero influx is `newborn_rate`, placed in the unvaccinated group and first disease cell. Default is 0.01 population units per year. There is no external adult inflow. The initial condition is a full `p_k(0,a,x)`, repairing the age omission in the manuscript's initial-condition notation.
- **Diffusion is set to zero.** The displayed PDE has no diffusion term, although the incidence formula contains `D_k`. A nonzero `D_k` would require adding diffusion and specifying corresponding boundary conditions; it is not silently implemented here.
- Cancer incidence is the age integral of outward x-flux, summed over groups. The paper's abbreviated incidence notation omits age integration.
- For runnable vaccination controls, `R_i=h_i p_6` and `R_6=-sum_i h_i p_6`. These are per-capita hazards, an explicit specialization of the unspecified `R_k`. They conserve population and cannot vaccinate more people than available under the CFL bound. In the paper's controlled notation the additive source is `u^2_i(t,a,x)=h_i p_6(t,a,x)`; **h_i is not u^2_i**.
- Treatment effort and hazards are constant in this implementation. To implement time-dependent policies, change the policy evaluation in `velocity`/`step` and ensure steps end at policy discontinuities.
- New vaccination transfers are age restricted: one dose before 12 and two doses before 12 require age <12; older one dose requires age >=12; two doses 12–18 uses 12<=age<18; two doses 18–26 uses 18<=age<27. Age-cell centers determine eligibility. The default one-year grid aligns threshold faces. Vaccination-history group membership persists as people age. Hazards still represent direct effective entry from unvaccinated to a status, with no sequential dosing or dose delay. No additional minimum vaccination age is imposed.
- The code simulates specified controls; it does not solve the optional PDE optimal-control problem or enforce the paper's cost budget.

Population balance is `N(t) + cumulative_cancer + cumulative_age_outflow + cumulative_natural_deaths = N(0) + cumulative_births`. No prevalence normalization `I/N` is introduced. If the population is rescaled from a fraction to counts by a factor c, beta must be divided by c to preserve dynamics.

## Parameters and inputs whose values are integral to the model

The manuscript specifies functions rather than a complete set of scalar coefficients. The following inputs must be identified or calibrated. Rates refer to a chosen common time unit (years here).

| Manuscript input | Role / units | Python placeholder or required decision |
|---|---|---|
| `beta(a)` | Infection coefficient, disease coordinate / (population unit × year) | Gaussian profile: `beta_peak=0.12`, `beta_age_center=27`, `beta_age_width=15`; replace `self.beta` for other age patterns |
| `nu_k(a)`, six functions | Restorative/clearance drift, disease coordinate/year | `clearance=(0.10,0.08,0.06,0.04,0.025,0.01)` divided by `1+a/clearance_age_scale`, with scale 80 years |
| `gamma(a)` | Treatment effect on drift per unit effort | Constant `gamma=0.05`; replace with an age array if supported by data |
| `u1(t,...)` | Nonnegative screening/treatment effort | `treatment_effort=0` |
| `R_k` or `u^2_k` | Vaccination source density per year, with zero sum across groups | `vaccination_portion_per_year=(0,0,0,0,0)` per year; multiply by unvaccinated density; age eligibility is enforced; sequential dose rules still require specification |
| `p_k(0,a,x)` | Initial joint age/disease distribution for each group | Full cell-average array, or the illustrative profile parameters below |
| `N(0)` | Initial population scale | `initial_population=1`; must use consistent beta scaling |
| Initial group masses | Vaccination-status proportions | `initial_group_fractions=(.02,.02,.02,.02,.02,.90)` |
| Initial age profile | Age composition within each group | Exponential with `initial_age_scale=35`; same age profile for all groups |
| Initial disease profile | Distribution within each group | Truncated Gaussian with `initial_x_means=(.08,.10,.12,.14,.16,.25)` and `initial_x_width=.08`; grid-normalized, so actual means differ near boundaries |
| Age-zero boundary | Recruitment rate and distribution over groups/disease | `newborn_rate=0.01` population units/year; enters unvaccinated first disease cell |
| Disease boundaries | Health retention and cancer exit behavior | Zero healthy flux and absorbing cancer outflow, as explained above |
| `D_k` if diffusion intended | Disease-coordinate squared/year | Fixed at zero because absent from the evolution equation; requires a revised PDE to enable |
| `T` | Simulation duration, years | `--years 10` |
| `q1`, `q^2_i`, `M` (optimization only) | Treatment cost, five vaccination costs, available budget with consistent units | Not used for prescribed-policy simulation; required to formulate and solve the paper's constrained PDE optimization |

The manuscript orders `nu_0 > ... > nu_5` even though the groups elsewhere are indexed 1–6. The code uses the listed six groups in order (Python indices 0–5) and decreasing baseline drifts. The statement “nu=1 indicates full immunity” is not mathematically guaranteed by `v=beta I-nu`: a calibrated rate scale and bound on beta I would also be needed. The unusual ordering between one-dose and two-dose groups follows the manuscript's ordering, and needs scientific review before calibration.

Numerical settings are not biological parameters: `max_age=100`, `n_age=100`, `n_x=100`, `cfl=.8`, and `--save-every 1`. Choose an age cutoff consistent with the modeled population and assess truncation and grid sensitivity.

## Run in Anaconda / Spyder

Open `run_spyder.py` in Spyder and press F5 (Run file). Keep `hpv_pde.py` in the same folder. Edit the settings and `Parameters(...)` block near the top of the launcher, then rerun the entire file. This launcher takes no command-line arguments and finds its sibling solver independently of Spyder's working directory.

The arrays `time`, `density`, `summary`, `age`, `x`, `population`, `burden`, and `cumulative_cancer` remain available in Variable Explorer. `density` is a four-dimensional array; inspect a slice such as `density[-1, 5, :, :]` in the console for the final unvaccinated age/disease distribution. Three summary plots appear using Spyder's configured Matplotlib display. Set `MAKE_PLOTS=False` to disable plotting. The solution is saved beside the launcher under `spyder_results/solution.npz`. NumPy is required; plotting additionally needs Matplotlib.

## Plot images and saved file formats

Running `run_spyder.py` with `MAKE_PLOTS=True` displays five figures and saves them under `spyder_results/plots`: overview, vaccination-group populations, initial/final disease and age distributions, initial/final age–disease heatmaps, and cancer incidence with population accounting. Both heatmaps use the same color scale. Incidence is the average cancer outflow over each saved time interval. With the illustrative defaults cancer outflow can be very small; the vertical axis may use scientific notation. A disease coordinate is not a prevalence estimate.

`solution.npz` is a compressed NumPy archive containing the numerical arrays (including the full time/age/disease/group solution). It lets you reload results without rerunning the simulation. For example:

```python
import numpy as np
with np.load('spyder_results/solution.npz') as saved:
    time = saved['time']
    density = saved['density']
```

`parameters.json` is a human-readable record of the parameter values used for that run. It contains settings, not the solution. `example_parameters.json` is an input configuration you may edit for the command-line solver. The Spyder launcher takes its settings from its `Parameters(...)` block and writes `parameters.json` as a record; editing that output JSON does not change the Spyder run.

To plot an existing output produced by the command-line solver, run `python3 plot_hpv.py results/solution.npz`. Older archives without `summary` still support population/distribution figures; rerun the solver to include incidence/accounting in those archives.


## Exponential natural mortality extension

At the user's request the model now includes `-mu(a)*p_k` on the right-hand side, with

    mu(a) = mortality_at_birth * exp(mortality_growth * a).

Defaults are `mortality_at_birth=0.0001` per year and `mortality_growth=0.08` per year of age. These are illustrative, uncalibrated values, giving hazards of about 0.00050/year at 20, 0.00546/year at 50, and 0.0602/year at 80. These are instantaneous hazards, not annual death probabilities. At fixed age the annual death probability corresponding to hazard mu is `1-exp(-mu)`; actual people continue aging. This simple monotonically increasing hazard does not model an infant mortality peak.

Edit both parameters in `run_spyder.py`. Set `mortality_at_birth=0` to recover the previous no-natural-mortality behavior. Mortality affects all disease states and vaccination groups equally. The combined CFL bound now includes the largest mortality hazard to preserve nonnegative densities. Natural deaths are integrated separately from cancer exits and the finite maximum-age boundary. Cancer exits leave the tracked population; their later survival is not modeled.

`summary[:,18]` (zero-based indexing) and CSV column `cumulative_natural_deaths` record cumulative natural deaths. Existing columns retain their positions. The Spyder launcher exposes `cumulative_natural_deaths` and `mortality_by_age`. Natural deaths appear in the population accounting figure, and `06_mortality_by_age.png` shows the age-dependent hazard. The revised mass balance includes births and all three types of exit.


## Age/time and cohort views

`07_age_time_by_group.png` shows six time-versus-age heatmaps colored by mean disease coordinate (common 0–1 scale), with diagonal aging paths. Gray cells have negligible population. `08_cohort_progression.png` shows moving ten-year age-band means for initial ages 0–9 through 70–79, in six vaccination-group panels. Curves stop when their full age band crosses the maximum-age cutoff. These approximate cohort views condition on remaining in the model/current group, and are affected by cancer exits, deaths, vaccination transfers, and numerical mixing; they are not individual trajectories. Both are automatically displayed and saved when running `run_spyder.py`.


## Census demographic initialization (October 24, 2026)

The Spyder launcher uses U.S. resident population, both sexes, July 1, 2025 (Census Vintage 2025). At max_age=100 the model represents ages 0–99: 341,665,675 people. The open-ended 100+ count (119,182) is excluded, not distributed arbitrarily. The prepared age CSV replaces the exponential age profile. Its single-year counts are conservatively remapped if the grid changes. The loader rejects a cutoff above the data's upper age of 100.

Annual births of 3,620,461 (July 2024–July 2025, Census NC-EST2025-COMPN!I7) give `newborn_rate = initial_population * 3620461 / represented_Census_population`, about 0.010596502 for initial_population=1 and max_age=100. This is constant annual influx, not lifetime total fertility or a changing-population percentage. Edit ANNUAL_BIRTHS, INITIAL_POPULATION, MAX_AGE, N_AGE and INITIAL_AGE_DATA_FILE in run_spyder.py; every model parameter remains visible there. Set initial_age_data_file=None to restore the exponential fallback.

The article's linked tables now include subsequent updates: the births workbook is a June 2026 release. Source URLs, dates, counts, and hashes are in census_data/sources.json. Raw files and prepared counts ship locally; model runs require no internet or spreadsheet-reading library. Demographic provenance is saved alongside each Spyder run.

This calibrates the initial national age marginal and baseline births only. Vaccination/disease profiles and mortality are still assumptions, and migration is absent. Calendar time zero is July 1, 2025.


## Additional figures 10–14

Run run_spyder.py with F5 to display/save annual births and all losses (10), stacked vaccination-group proportions (11), disease-distribution heatmaps through time (12), vaccination scenario comparisons (13), and initially present versus later-born populations (14). Annual flows are averages between output snapshots. Density heatmaps use a common absolute scale, not within-group normalization.

The launcher exposes TRACK_INITIAL_POPULATION, RUN_SCENARIOS, and SCENARIO_POLICIES. Default comparisons use per-target rates 0, .05, and .5 per year. They share all other settings and the same initial state; no-new-vaccination retains the initially vaccinated population. Matching current policies reuse the existing run. The tracer follows the initially present population using the total population's transmission pressure and no births, leaving the full solution unchanged. Upwind age diffusion still blurs cohort fronts.

The archive solution.npz additionally stores initial_cohort_age_density with axes [time, age] when tagging is enabled. scenario_comparison.npz and scenario_policies.json store comparison results/settings. Turn RUN_SCENARIOS=False to avoid extra simulation work; turn TRACK_INITIAL_POPULATION=False to disable cohort tracking.
