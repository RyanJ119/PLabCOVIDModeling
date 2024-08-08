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
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.plot(t, series[:, 0])
    plt.ylim(top=1.5)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-.5)  # adjust the bottom leaving top unchanged
    plt.title(
        f"Lockdown of Public R0={problem.R0} "
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
    
    
def print_heat_map(
         directory_path, w, S, E, I, R, beta, percentage_essential, problem, show=False
):
    return None


def generate_all_plots(directory_path, w, S, E, I, R,cost, beta, percentage_essential,cost_of_lockdown, problem, show=False, store_data=False):
    if store_data:
        try:
            pd.DataFrame(np.array(S)).to_csv('../Starting point/S.csv', index=False)
            pd.DataFrame(np.array(E)).to_csv('../Starting point/E.csv', index=False)
            pd.DataFrame(np.array(I)).to_csv('../Starting point/I.csv', index=False)
            pd.DataFrame(np.array(R)).to_csv('../Starting point/R.csv', index=False)
            pd.DataFrame(np.array(w)).to_csv('../Starting point/w.csv', index=False)
        except:
            try:
                pd.DataFrame(S).to_csv('../Starting point/S.csv', index=False)
                pd.DataFrame(E).to_csv('../Starting point/E.csv', index=False)
                pd.DataFrame(I).to_csv('../Starting point/I.csv', index=False)
                pd.DataFrame(R).to_csv('../Starting point/R.csv', index=False)
                pd.DataFrame(w).to_csv('../Starting point/w.csv', index=False)
            except:
                pd.DataFrame(np.array(list(S))).to_csv('../Starting point/S.csv', index=False)
                pd.DataFrame(np.array(list(E))).to_csv('../Starting point/E.csv', index=False)
                pd.DataFrame(np.array(list(I))).to_csv('../Starting point/I.csv', index=False)
                pd.DataFrame(np.array(list(R))).to_csv('../Starting point/R.csv', index=False)
                pd.DataFrame(np.array(list(w))).to_csv('../Starting point/w.csv', index=False)

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


