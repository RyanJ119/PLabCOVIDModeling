import numpy as np
from utils import transform_to_have_essential_workers, read_data_from_csv, make_result_directory_for_simulation, Problem
from plotting import generate_all_plots
from plotting import print_heat_map
from simulator import simulate
#from controller import solve_control_problem
from ControllerCanibalizedW import solve_control_problem
from controllerV2 import solve_control_problem
import csv

import time

def _last_positive_component_in_group(S, idx):
    indices = [i for i, s in enumerate(S[:, idx]) if s > 0.0]
    if len(indices) > 0:
        return max(indices) - 1
    return len(S[:, idx])




def main():
    state_id = "FL"
    data = read_data_from_csv("../data/covid19_data_v1.csv")
    state_data = data[state_id]
    death_rates = np.array(
        [
            [
                0.00016,
                0.00016,
                0.00006,
                0.00006,
                0.00007,
                0.00007,
                0.00229,
                0.00229,
                0.01915,
                0.01915,
                0.13527,
            ]
        ]
    )
    cost_lockdown = 70
    num_age_groups = len(death_rates[0])
    time_horizon = 180
    #matrices = [(0.25, "../data/InteractionMatrix_beta_0_25.csv"), (0.5, "../data/InteractionMatrix_beta_0_5.csv"), (0.75,  "../data/InteractionMatrix_beta_0_75.csv")]
    matrices = [(0.75,  "../data/InteractionMatrix_beta_0_75.csv")]
    #R0s = [ 1.0, 1.1, 1.2]
    R0s = [1.2]
   # percentages_essential = [0.24, 0.34, 0.44]
    percentages_essential = [0]
    ordering = np.flip(list(range(num_age_groups)))
    max_num_vaccines_per_day = np.sum(state_data["initial_S"]) * 0.6 / time_horizon
    
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
                    #initial_V = np.zeros((1, num_age_groups))
                    

                    
                    
                    problem = Problem(
                        time_horizon,
                        R0,
                        initial_S,
                        initial_E,
                        initial_I,
                        initial_R,
                        
                        
                        population,
                        contact_matrix,
                        death_rates,
                        cost_lockdown,
                        60
                    )
                   
        
    
                    
            
               
         
                    S, E, I, R, w, cost = solve_control_problem(problem, max_num_vaccines_per_day)
                    dir_path = make_result_directory_for_simulation(state_id, contact_matrix_pair[0], R0, percentage_essential, "opt_control_")
                   # print(w)
                    print(cost)
                    generate_all_plots(
                        dir_path, np.array(w), np.array(S), np.array(E), np.array(I), np.array(R),  contact_matrix_pair[0], percentage_essential, cost_lockdown, problem, False
                    )

                    

    
if __name__ == "__main__":
    main()
