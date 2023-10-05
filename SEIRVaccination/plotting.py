import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sb
import pandas as pd
import csv


def generate_abstact_plot(
        directory_path, name, cost_deaths, series, beta, percentage_essential, problem, show=False
):
    t = np.linspace(0, problem.time_horizon, problem.N + 1)
    plt.rcParams["figure.figsize"] = (8, 8)

    # plt.plot(t, series[:, 0], label="Age group 1")
    # plt.plot(t, series[:, 1], label="Age group 2")
    # plt.plot(t, series[:, 2], label="Age group 3")
    # plt.plot(t, series[:, 3], label="Age group 3 Essential")
    # plt.plot(t, series[:, 4], label="Age group 4")
    # plt.plot(t, series[:, 5], label="Age group 4 Essential")
    # plt.plot(t, series[:, 6], label="Age group 5")
    # plt.plot(t, series[:, 7], label="Age group 5 Essential")
    # plt.plot(t, series[:, 8], label="Age group 6")
    # plt.plot(t, series[:, 9], label="Age group 6 Essential")
    # plt.plot(t, series[:, 10], label="Age group 7")
    plt.plot(t, series[:, 0])
    plt.plot(t, series[:, 1])
    plt.plot(t, series[:, 2])
    plt.plot(t, series[:, 3])
    plt.plot(t, series[:, 4])
    plt.plot(t, series[:, 5])
    plt.plot(t, series[:, 6])
    plt.plot(t, series[:, 7])
    plt.plot(t, series[:, 8])
    plt.plot(t, series[:, 9])
    plt.plot(t, series[:, 10])
    plt.title(
        f"{name} R0{problem.R0} PE{percentage_essential} Cost: {cost_deaths}"
    )
   #age-{age_group}-virLoadProf.png
    #age-{age_group}-SIRDynamics.png
    plt.savefig(
        os.path.join(directory_path, f"beta-{beta}-R0-{problem.R0}-PE-{percentage_essential}-{name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()

def print_heat_map(
         directory_path, w, S, E, I, R, Sv, Iv, Ev, Rv, beta, percentage_essential, problem, show=False
):
    cost_deaths = np.sum(R[-1, :] * problem.death_rates)
    days = np.linspace(0, problem.time_horizon, problem.N + 1)

    vac_plan  = pd.DataFrame(np.array(w))

    groups = ['0-4', '5-14', '15-19 non-essential', '15-19 essential', '20-39 non-essential', 

              '20-39 essential', '40-59 non-essential','40-59 essential', '60-69 non-essential', 

              '60-69 essential','70+']

    plt.figure(figsize=(12, 12))

    

    sb.heatmap(np.transpose(np.array(vac_plan)), cmap='Blues', robust=True,

           xticklabels=[day if day % 5 == 0 or day == max(days) else '' for day in days])

    cax = plt.gcf().axes[-1]

    cax.tick_params(labelsize=12)

    plt.xticks(fontsize=18)

    plt.yticks(fontsize=18)

    plt.title(
        f"FL R0{problem.R0} PE{percentage_essential} Cost: {cost_deaths}", fontsize=2
    )
    plt.savefig(
        os.path.join(directory_path, f"beta-{beta}-R0-{problem.R0}-PE-{percentage_essential}-Heat-Map-Vaccine-Policy.png"),
        dpi=300,
        bbox_inches="tight",
    )
    
    if show:
        plt.show()
    plt.close()


def generate_all_plots(directory_path, w, S, E, I, R, Sv, Iv, Ev, Rv, beta, percentage_essential, problem, show=False):
    cost_deaths = np.sum(R[-1, :] * problem.death_rates)
    

    generate_abstact_plot(
        directory_path, "vaccines_policy", cost_deaths, w, beta, percentage_essential, problem, show
    )
    generate_abstact_plot(
                directory_path, "susceptible", cost_deaths, S, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "exposed", cost_deaths, E, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "infected", cost_deaths, I, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "recovered", cost_deaths, R, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "Vaccinated susceptible", cost_deaths, Sv, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "Vaccinated exposed", cost_deaths, Ev, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "Vaccinated infected", cost_deaths, Iv, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "Vaccinated recovered", cost_deaths, Rv, beta, percentage_essential, problem, show=False
    )
