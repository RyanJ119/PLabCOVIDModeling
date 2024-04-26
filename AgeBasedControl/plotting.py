import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sb
import pandas as pd
import csv

control_names=["Lockdown of Elderly", "Lockdown of School Age", "Lockdown of Public","Closure of public transports"]
path_names=["Old","School","GenPop","PubTransports"]
def generate_abstact_plot(
        directory_path, name, cost, series, beta, percentage_essential, problem, show=False
):
    t = np.linspace(0, problem.time_horizon, problem.N + 1)
    t=t[0:160]
    plt.rcParams["figure.figsize"] = (8, 8)

    plt.plot(t, series[:, 0], label="Age group 1")
    plt.plot(t, series[:, 1], label="Age group 2")
    plt.plot(t, series[:, 2], label="Age group 3")
    plt.plot(t, series[:, 4], label="Age group 4")
    plt.plot(t, series[:, 6], label="Age group 5")
    plt.plot(t, series[:, 8], label="Age group 6")
    plt.plot(t, series[:, 10], label="Age group 7", color='black')
    plt.title(
        f"{name}"
    )
    plt.legend()
    plt.savefig(
        os.path.join(directory_path, f"beta-{beta}-R0-{problem.R0}-PE-{percentage_essential}-{name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()
    
def generate_abstact_plotU(
        directory_path, name, cost, series, beta, percentage_essential, problem, show=False
):
    plt.rcParams.update({'font.size': 22})
    t = np.linspace(0, problem.time_horizon, problem.N + 1)
    t=t[0:160]
    for control in range(len(series[0,:])):
        plt.rcParams["figure.figsize"] = (8, 8)
        plt.plot(t, series[:, control])
        plt.ylim(top=1.5)  # adjust the top leaving bottom unchanged
        plt.ylim(bottom=-.5)  # adjust the bottom leaving top unchanged
        plt.title(
            f"{control_names[control]} R0={problem.R0}"
        )
        plt.legend()
        plt.savefig(
            os.path.join(directory_path, f"{path_names[control]}-beta-{beta}-R0-{problem.R0}-PE-{percentage_essential}-{name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close()

    
def print_heat_map(
         directory_path, w, S, E, I, R, beta, percentage_essential, problem, show=False
):
    cost_deaths = np.sum(R[-1, :] * problem.death_rates)
    
    days = np.linspace(0, problem.time_horizon, problem.N + 1)

    vac_plan  = pd.DataFrame(np.array(w))

    groups = ['0-4', '5-14', '15-19 non-essential', '15-19 essential', '20-39 non-essential', 

              '20-39 essential', '40-59 non-essential','40-59 essential', '60-69 non-essential', 

              '60-69 essential','70+']

    plt.figure(figsize=(12, 12))

    

    sb.heatmap(np.transpose(np.array(vac_plan)), cmap='Blues', robust=True,

           xticklabels=[day if day % 5 == 0 or day == max(days) else '' for day in days],

           yticklabels=groups)

    cax = plt.gcf().axes[-1]

    cax.tick_params(labelsize=12)

    plt.xticks(fontsize=12)

    plt.yticks(fontsize=14)

    plt.title(
        f"Cost: {cost_deaths}"
    )
    plt.savefig(
        os.path.join(directory_path, f"beta-{beta}-R0-{problem.R0}-PE-{percentage_essential}-Heat-Map-Vaccine-Policy.png"),
        dpi=300,
        bbox_inches="tight",
    )
    
    if show:
        plt.show()
    plt.close()


def generate_all_plots(directory_path, w, S, E, I, R,cost, beta, percentage_essential,cost_of_lockdown, problem, show=False):
    generate_abstact_plotU(
        directory_path, "Lockdown Policy", cost, w, beta, percentage_essential, problem, show
    )
    generate_abstact_plot(
                directory_path, "Susceptible Per Day", cost, S, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "Exposed Per Day", cost, E, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "Infected Per Day", cost, I, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "Recovered Per Day", cost, R, beta, percentage_essential, problem, show=False
    )


