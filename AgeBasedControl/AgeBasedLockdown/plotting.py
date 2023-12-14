import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sb
import pandas as pd
import csv


def generate_abstact_plot(
        directory_path, name, cost, series, beta, percentage_essential, problem, show=False
):
    t = np.linspace(0, problem.time_horizon, problem.N + 1)
    plt.rcParams["figure.figsize"] = (8, 8)

    plt.plot(t, series[:, 0], label="Age group 1")
    plt.plot(t, series[:, 1], label="Age group 2")
    plt.plot(t, series[:, 2], label="Age group 3")
    plt.plot(t, series[:, 3], label="Age group 3 Essential")
    plt.plot(t, series[:, 4], label="Age group 4")
    plt.plot(t, series[:, 5], label="Age group 4 Essential")
    plt.plot(t, series[:, 6], label="Age group 5")
    plt.plot(t, series[:, 7], label="Age group 5 Essential")
    plt.plot(t, series[:, 8], label="Age group 6")
    plt.plot(t, series[:, 9], label="Age group 6 Essential")
    plt.plot(t, series[:, 10], label="Age group 7", color='black')
    plt.title(
        f"{name} R0{problem.R0} PE{percentage_essential} Cost: {cost}"
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
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.plot(t, series[:, 0])
    plt.ylim(top=1.5)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-.5)  # adjust the bottom leaving top unchanged
    plt.title(
        f"Lockdown of Elderly R0-{problem.R0}"
    )
    plt.legend()
    plt.savefig(
        os.path.join(directory_path, f"Old-beta-{beta}-R0-{problem.R0}-PE-{percentage_essential}-{name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()
    
    t = np.linspace(0, problem.time_horizon, problem.N + 1)
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.plot(t, series[:, 1])
    plt.ylim(top=1.5)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-.50)  # adjust the bottom leaving top unchanged
    plt.title(
        f"Lockdown of School Age R0{-problem.R0}"
    )
    plt.legend()
    plt.savefig(
        os.path.join(directory_path, f"School-beta-{beta}-R0-{problem.R0}-PE-{percentage_essential}-{name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()
    
    
    t = np.linspace(0, problem.time_horizon, problem.N + 1)
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.plot(t, series[:, 2])
    plt.ylim(top=1.5)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-.50)  # adjust the bottom leaving top unchanged
    plt.title(
        f"Lockdowng of Public R0{problem.R0} "
    )
    plt.legend()
    plt.savefig(
        os.path.join(directory_path, f"GenPop-beta-{beta}-R0-{problem.R0}-PE-{percentage_essential}-{name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()


    # t = np.linspace(0, problem.time_horizon, problem.N + 1)
    # plt.rcParams["figure.figsize"] = (8, 8)
    # plt.plot(t, series[:, 0]+(series[:, 2]+series[:, 1]), label="public plus schools")
    # plt.ylim(top=1.5)  # adjust the top leaving bottom unchanged
    # plt.ylim(bottom=-.50)  # adjust the bottom leaving top unchanged
    # plt.title(
    #     f"{name} R0{problem.R0} PE{percentage_essential} Cost: {cost}"
    # )
    # plt.legend()
    # plt.savefig(
    #     os.path.join(directory_path, f"public_Plus_School-beta-{beta}-R0-{problem.R0}-PE-{percentage_essential}-{name}.png"),
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # if show:
    #     plt.show()
    # plt.close()
    
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
        f"FL R0{problem.R0} PE{percentage_essential} Cost: {cost_deaths}"
    )
    plt.savefig(
        os.path.join(directory_path, f"beta-{beta}-R0-{problem.R0}-PE-{percentage_essential}-Heat-Map-Vaccine-Policy.png"),
        dpi=300,
        bbox_inches="tight",
    )
    
    if show:
        plt.show()
    plt.close()


def generate_all_plots(directory_path, w, S, E, I, R, beta, percentage_essential,cost_of_lockdown, problem, show=False):
    #cost_deaths = np.sum(R[-1, :] * problem.death_rates)
    #cost_lockdown= np.sum(((1-u))*cost_of_lockdown*S[-1, :])
    cost=1 #cost_deaths+cost_lockdown
    generate_abstact_plotU(
        directory_path, "Lockdown Policy", cost, w, beta, percentage_essential, problem, show
    )
    generate_abstact_plot(
                directory_path, "susceptible", cost, S, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "exposed", cost, E, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "infected", cost, I, beta, percentage_essential, problem, show=False
    )
    generate_abstact_plot(
                directory_path, "recovered", cost, R, beta, percentage_essential, problem, show=False
    )


