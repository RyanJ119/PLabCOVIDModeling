"""Plot saved HPV PDE results or call plot_results from Spyder.
Run: python3 plot_hpv.py results/solution.npz
"""
from pathlib import Path
import numpy as np

LABELS = ('One dose <12', 'One dose >12', 'Two doses <12',
          'Two doses 12–18', 'Two doses 18–26', 'Unvaccinated')

def plot_results(time, age, x, density, summary=None, output_folder='plots', show=True, mortality=None,
                 scenarios=None, initial_cohort_age_density=None):
    import matplotlib.pyplot as plt
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Input density axes are [time, group, age, disease]. Summing removes axes;
    # multiply by the corresponding cell widths to integrate densities into population.
    da, dx = age[1]-age[0], x[1]-x[0]

    group_population = density.sum(axis=(2, 3))*da*dx
    population = group_population.sum(axis=1)
    burden = (density*x[None, None, None, :]).sum(axis=(1, 2, 3))*da*dx

    # Mean disease coordinate = burden / tracked population; this is not prevalence.
    mean_x = np.divide(burden, population, out=np.zeros_like(burden), where=population > 0)
    figures = []

    # Each figure is saved as PNG and returned for interactive use in Spyder.
    def save(fig, name):
        fig.savefig(output_folder/name, dpi=180, bbox_inches='tight')
        figures.append(fig)

    # Figure 01: population, burden, mean disease coordinate, and cumulative cancer.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    series = [(population, 'Tracked population'), (burden, 'Total disease burden'),
              (mean_x, 'Mean disease coordinate among tracked people')]
    for ax, (values, label) in zip(axes.flat, series):
        ax.plot(time, values)
        ax.set(xlabel='Time (years)', ylabel=label)
        ax.grid(alpha=0.3)
    # Figure 01 has no cancer panel if a saved file lacks the summary array.
    if summary is not None:
        axes[1, 1].plot(time, summary[:, 3])
        axes[1, 1].set(xlabel='Time (years)', ylabel='Cumulative cancer outflow')
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].set_visible(False)
    fig.suptitle('HPV PDE results — illustrative parameters')
    save(fig, '01_overview.png')

    # Figure 02: absolute population by vaccination-history group.
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for k, label in enumerate(LABELS):
        ax.plot(time, group_population[:, k], label=label)
    ax.set(xlabel='Time (years)', ylabel='Population in group', title='Vaccination groups')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, '02_vaccination_groups.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    # Figure 03: compare the first and last snapshots, including later births.
    for idx, label in ((0, 'Initial'), (-1, 'Final')):
        axes[0].plot(x, density[idx].sum(axis=(0, 1))*da, label=label)
        axes[1].plot(age, density[idx].sum(axis=(0, 2))*dx, label=label)
    axes[0].set(xlabel='Disease coordinate x (0=healthy, 1=cancer)', ylabel='Population density per unit x')
    axes[1].set(xlabel='Age (years)', ylabel='Population density per year of age')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle('Initial and final distributions (all groups combined)')
    save(fig, '03_distributions.png')

    if summary is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

        # Figure 05: cumulative increments / elapsed time give average cancer rates.
        interval_incidence = np.diff(summary[:, 3])/np.diff(time)
        axes[0].stairs(interval_incidence, time)
        axes[0].set(xlabel='Time (years)', ylabel='Cancer outflow per year',
                    title='Average cancer incidence per output interval')
        axes[1].plot(time, population, label='Tracked population')
        axes[1].plot(time, summary[:, 3], label='Cumulative cancer exits')
        axes[1].plot(time, summary[:, 4], label='Cumulative age exits')
        deaths = summary[:, 18] if summary.shape[1] >= 19 else np.zeros_like(time)
        axes[1].plot(time, deaths, label='Cumulative natural deaths')

        # This should remain N(0): remaining population + exits - births.
        balance = population+summary[:, 3]+summary[:, 4]+deaths-summary[:, 5]
        axes[1].plot(time, balance, '--', label='Population + exits − births')
        axes[1].set(xlabel='Time (years)', ylabel='Population units', title='Population accounting')
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.3)
        save(fig, '05_incidence_and_balance.png')
    # Figure 06: natural mortality hazard by age, not an annual death probability.
    if mortality is not None:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(age, mortality)
        ax.set(xlabel='Age (years)', ylabel='Natural mortality hazard (per year)',
               title='Exponential age-dependent mortality — illustrative parameters')
        ax.grid(alpha=0.3)
        save(fig, '06_mortality_by_age.png')

    # Figure 07: mean disease coordinate at each time, group, and age.
    # Mask near-empty populations instead of assigning them a misleading healthy mean.
    age_mass = density.sum(axis=3)*dx
    age_burden = (density*x[None, None, None, :]).sum(axis=3)*dx

    threshold = max(float(age_mass.max())*1e-10, np.finfo(float).tiny)
    age_mean = np.divide(age_burden, age_mass,
                         out=np.full_like(age_mass, np.nan), where=age_mass > threshold)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True,
                             sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):

        # Transpose [time,age] to [age,time] for vertical age and horizontal time.
        # All panels share the same 0--1 disease-mean color scale.
        im = ax.pcolormesh(time, age, np.ma.masked_invalid(age_mean[:, k, :].T),
                           shading='nearest', vmin=0, vmax=1, cmap='viridis')
        ax.set_facecolor('lightgray')
        for start_age in (10, 30, 50, 70):

            # Dashed lines are age reference paths, not separately tagged people.
            path_age = start_age+time-time[0]
            visible = path_age <= age[-1]+da/2
            ax.plot(time[visible], path_age[visible], 'w--', lw=0.9, alpha=0.8)
        ax.set(title=LABELS[k], xlabel='Time (years)', ylabel='Age (years)',
               ylim=(0, age[-1]+da/2))
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Mean disease coordinate (0–1)')
    fig.suptitle('Disease by age and time — dashed paths follow aging cohorts\n'
                 'Gray: negligible population; means describe people still tracked')
    save(fig, '07_age_time_by_group.png')

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True,
                             sharex=True, sharey=True)

    # Figure 08: approximate ten-year birth-cohort bands move forward in age.
    # These are conditional averages, affected by exits and vaccination transfers.
    starts = (0, 10, 20, 30, 40, 50, 60, 70)
    for k, ax in enumerate(axes.flat):
        for start in starts:
            lo = start+time-time[0]
            hi = lo+10

            # Fractional cell overlap integrates over the moving band without snapping
            # the whole band to the nearest grid cells.
            weights = np.maximum(0, np.minimum(age[None, :]+da/2, hi[:, None])
                                 - np.maximum(age[None, :]-da/2, lo[:, None]))
            mass = (age_mass[:, k, :]*weights).sum(axis=1)
            weighted_burden = (age_burden[:, k, :]*weights).sum(axis=1)

            mean = np.divide(weighted_burden, mass, out=np.full_like(mass, np.nan),
                             where=mass > threshold*10)

            # Stop before the band is partly outside the model's maximum-age domain.
            mean[hi > age[-1]+da/2+1e-10] = np.nan
            ax.plot(time, mean, label=f'{start}–{start+9} initially')
        ax.set(title=LABELS[k], xlabel='Time (years)', ylabel='Mean disease coordinate',
               ylim=(0, 1))
        ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8)
    fig.suptitle('Approximate cohort progression in moving age bands\n'
                 'Conditional on remaining in the model and current vaccination group')
    save(fig, '08_cohort_progression.png')

    # Age marginal reused for the tagged-cohort comparison in Figure 14.
    population_by_age = density.sum(axis=(1, 3))*dx

    # Figure 10: interval-average births, exits, and net change (not cumulative totals).
    if summary is not None and summary.shape[1] >= 19:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        rates = {}
        for label, column in (('Births', 5), ('Natural deaths', 18),
                              ('Cancer exits', 3), ('Maximum-age exits', 4)):
            rates[label] = np.diff(summary[:, column])/np.diff(time)
            ax.stairs(rates[label], time, label=label, linewidth=1.8, baseline=None)
        net = rates['Births']-rates['Natural deaths']-rates['Cancer exits']-rates['Maximum-age exits']
        ax.stairs(net, time, label='Net population change', linestyle='--', color='black', baseline=None)
        ax.axhline(0, color='gray', lw=0.7)
        ax.set(xlabel='Time (years)', ylabel='Population units per year',
               title='Births and losses — averages over saved time intervals')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        save(fig, '10_births_and_losses_per_year.png')

    # Figure 11: fractions of people currently tracked; each time slice sums to 1.
    fractions = np.divide(group_population, population[:, None],
                          out=np.full_like(group_population, np.nan),
                          where=population[:, None] > 0)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.stackplot(time, fractions.T, labels=LABELS)
    ax.set(xlabel='Time (years)', ylabel='Fraction of tracked population', ylim=(0, 1),
           title='Vaccination composition over time')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    save(fig, '11_vaccination_composition.png')

    from matplotlib.colors import PowerNorm
    # Figure 12: integrate over ages, retaining time, group, and disease.
    # Color is density per unit x, which may exceed 1. A cell contains density*dx;
    # 100*density*dx/initial_population converts it to percent of initial population.
    disease_density = density.sum(axis=2)*da
    vmax = max(float(disease_density.max()), np.finfo(float).tiny)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True,
                             sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        im = ax.pcolormesh(time, x, disease_density[:, k, :].T,
                           shading='nearest', cmap='viridis',
                           # Power normalization reveals low densities while retaining a shared scale.
                           # It changes only color contrast, not density values or the simulation.
                           norm=PowerNorm(gamma=0.3, vmin=0, vmax=vmax))
        ax.set(xlabel='Time (years)', ylabel='Disease coordinate x', title=LABELS[k],
               ylim=(0, 1))
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Population density per unit x')
    fig.suptitle('Disease distributions through time — shared power color scale (gamma=0.3)')
    save(fig, '12_disease_distributions_over_time.png')

    # Figure 13: compare illustrative uptake policies with identical other settings.
    if scenarios:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for result in scenarios:
            axes[0].plot(time, result['summary'][:, 2], label=result['label'])
            axes[1].plot(time, result['summary'][:, 3], label=result['label'])
        for ax, ylabel in zip(axes, ('Total disease burden', 'Cumulative cancer exits')):
            ax.set(xlabel='Time (years)', ylabel=ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle('Vaccination scenarios — identical initial state and other parameters\n'
                     'Illustrative uptake policies; no optimization or cost adjustment')
        save(fig, '13_vaccination_scenarios.png')

    # Figure 14: the solver's passive tag isolates people present at the start.
    # Subtracting their age density from total age density gives later-born survivors.
    if initial_cohort_age_density is not None:

        original = np.asarray(initial_cohort_age_density)
        if original.shape != population_by_age.shape:
            raise ValueError('Initial-cohort age density must have shape (time, age)')
        later_births = population_by_age-original
        tolerance = max(float(population_by_age.max()), 1.0)*1e-10
        if later_births.min() < -tolerance:
            raise ValueError('Tagged initial population exceeds total population')
        # Clip only negligible negative roundoff after checking for larger discrepancies.
        later_births = np.maximum(later_births, 0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for idx in np.unique(np.linspace(0, len(time)-1, min(5, len(time)), dtype=int)):
            axes[0].plot(age, original[idx], label=f'{time[idx]:g} years')
        axes[0].set(xlabel='Age (years)', ylabel='Population density per year of age',
                    title='Only people present at the start — survivors at each time')
        axes[0].legend(fontsize=8)
        axes[1].stackplot(time, original.sum(axis=1)*da, later_births.sum(axis=1)*da,
                          labels=('Present at start', 'Born after start'))
        axes[1].set(xlabel='Time (years)', ylabel='Tracked population units',
                    title='Population by entry cohort')
        axes[1].legend(fontsize=8)
        fig.suptitle('Initial population versus later births — shared full-model dynamics')
        save(fig, '14_initial_population_and_births.png')

    # Spyder displays all figures; show=False saves images without opening windows.
    if show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
    return figures

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution', type=Path)
    args = parser.parse_args()
    with np.load(args.solution) as data:
        plot_results(data['time'], data['age'], data['x'], data['density'],
                     data['summary'] if 'summary' in data else None,
                     args.solution.parent/'plots',
                     mortality=data['mortality_by_age'] if 'mortality_by_age' in data else None,
                     initial_cohort_age_density=data['initial_cohort_age_density']
                     if 'initial_cohort_age_density' in data else None)
