import numpy as np
import csv

from constants import data, state_id, death_rates, num_age_groups, matrices, time_horizon, cost_lockdown, R0s, percentages_essential
from utils import transform_to_have_essential_workers, make_result_directory_for_simulation, Problem
from plotting import generate_all_plots
from controller_V2 import ProblemSolver2

def main():
    """Defines the Model and computes the optimal control associated"""
    state_data = data[state_id]

    with open('FL-Deaths.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "baseline", "descent", "descent Es 5", "opt", "simOpt", "evenly spaced"])

        for R0 in R0s:
            for contact_matrix_pair in matrices:
                contact_matrix = np.loadtxt(open(contact_matrix_pair[1], "rb"), delimiter=",")
                for percentage_essential in percentages_essential:
                    initial_S = transform_to_have_essential_workers(
                        state_data["initial_S"], percentage_essential=percentage_essential
                    )
                    initial_E = transform_to_have_essential_workers(
                        state_data["initial_E"], percentage_essential=percentage_essential
                    )
                    initial_I = transform_to_have_essential_workers(
                        state_data["initial_I"], percentage_essential=percentage_essential
                    )
                    initial_R = transform_to_have_essential_workers(
                        state_data["initial_R"], percentage_essential=percentage_essential
                    )
                    population = transform_to_have_essential_workers(
                        state_data["population"], percentage_essential=percentage_essential
                    )

                    problem = Problem(
                        time_horizon,
                        R0,
                        initial_S,
                        initial_E*.6,
                        initial_I*.6,
                        initial_R,


                        population,
                        contact_matrix,
                        death_rates,
                        cost_lockdown,
                        80
                    )

                    model=ProblemSolver2(problem)

                    S, E, I, R, w, cost = model.solve_control_problem()
                    dir_path = make_result_directory_for_simulation(model.model_name,state_id, contact_matrix_pair[0], R0, percentage_essential, "opt_control_")
                    print('deaths:')
                    print(sum(sum(np.array(R)[-1,:]*death_rates)))
                    generate_all_plots(
                        dir_path, np.array(w), np.array(S), np.array(E), np.array(I), np.array(R), cost,  contact_matrix_pair[0], percentage_essential, cost_lockdown, problem, False
                    )




if __name__ == "__main__":
    main()
