import numpy as np
from utils import transform_to_have_essential_workers, read_data_from_csv, make_result_directory_for_simulation, Problem
from plotting import generate_all_plots
from plotting import print_heat_map
from simulator import simulate
from controller import solve_control_problem
import csv


def _last_positive_component_in_group(S, idx):
    indices = [i for i, s in enumerate(S[:, idx]) if s > 0.0]
    if len(indices) > 0:
        return max(indices) - 1
    return len(S[:, idx])


def learn_optimal_control_fixed_order(problem, ordering, max_num_vaccines_per_day):
    if sorted(ordering) != sorted(range(len(ordering))):
        return
    N = problem.N
    num_age_groups = problem.num_age_groups
    w = np.zeros((N + 1, num_age_groups))
    i = 0
    breaking_point = 0

    while i < len(ordering):
        idx = ordering[i]
        w[breaking_point:, idx] = max_num_vaccines_per_day
        S, _, _, _, _, _ , _ , _  = simulate(problem, w)
        breaking_point = _last_positive_component_in_group(S, idx)
        w[breaking_point:, idx] = 0.0
        i += 1
    return w

def learn_optimal_control_fixed_order_5_E_First(problem, ordering, max_num_vaccines_per_day):
    if sorted(ordering) != sorted(range(len(ordering))):
        return
    N = problem.N
    num_age_groups = problem.num_age_groups
    w = np.zeros((N + 1, num_age_groups))
    i = 0
    breaking_point = 0
    idx = ordering[3]
    w[breaking_point:, idx] = max_num_vaccines_per_day
    S, _, _, _, _, _ , _ , _ = simulate(problem, w)
    breaking_point = _last_positive_component_in_group(S, idx)
    w[breaking_point:, idx] = 0.0

    while i < len(ordering):
        if idx !=4:

            idx = ordering[i]
            w[breaking_point:, idx] = max_num_vaccines_per_day
            S, _, _, _, _, _ , _ , _ = simulate(problem, w)
            breaking_point = _last_positive_component_in_group(S, idx)
        w[breaking_point:, idx] = 0.0
        i += 1

    return w

def learn_optimal_control_fixed_day(problem, ordering, max_num_vaccines_per_day):
    if sorted(ordering) != sorted(range(len(ordering))):
        return
    N = problem.N
    num_age_groups = problem.num_age_groups
    w = np.zeros((N + 1, num_age_groups))


    for i in range(180):

        w[0*int(N/num_age_groups):(1)*int(N/(num_age_groups-2)),10] = max_num_vaccines_per_day
        w[1*int(N/(num_age_groups-2)):(2)*int(N/(num_age_groups-2)),9] = max_num_vaccines_per_day
        w[2*int(N/(num_age_groups-2)):(3)*int(N/(num_age_groups-2)),8] = max_num_vaccines_per_day
        w[3*int(N/(num_age_groups-2)):(4)*int(N/(num_age_groups-2)),7] = max_num_vaccines_per_day
        w[4*int(N/(num_age_groups-2)):(5)*int(N/(num_age_groups-2)),6] = max_num_vaccines_per_day
        w[5*int(N/(num_age_groups-2)):(6)*int(N/(num_age_groups-2)),5] = max_num_vaccines_per_day
        w[6*int(N/(num_age_groups-2)):(7)*int(N/(num_age_groups-2)),4] = max_num_vaccines_per_day
        w[7*int(N/(num_age_groups-2)):(8)*int(N/(num_age_groups-2)),3] = max_num_vaccines_per_day
        w[8*int(N/(num_age_groups-2)):(9)*int(N/(num_age_groups-2)),2] = max_num_vaccines_per_day

    return w
def main():
    state_id = "NJ"
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
    num_age_groups = len(death_rates[0])
    time_horizon = 180
    matrices = [(0.25, "../data/InteractionMatrix_beta_0_25.csv"), (0.5, "../data/InteractionMatrix_beta_0_5.csv"), (0.75,  "../data/InteractionMatrix_beta_0_75.csv")]
    #matrices = [(0.25,  "../data/InteractionMatrix_beta_0_25.csv")]
   # R0s = [ 1.0, 1.1, 1.2]
    R0s = [1.2]
    #percentages_essential = [0.24, 0.34, 0.44]
    percentages_essential = [0.44]
    ordering = np.flip(list(range(num_age_groups)))
    max_num_vaccines_per_day = np.sum(state_data["initial_S"]) * 0.6 / time_horizon

    with open('NJ-Deaths.csv', 'w', newline='') as file:
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

                    initial_Sv = np.zeros((1, num_age_groups))
                    initial_Ev = np.zeros((1, num_age_groups))
                    initial_Iv = np.zeros((1, num_age_groups))
                    initial_Rv = np.zeros((1, num_age_groups))


                    problem = Problem(
                        time_horizon,
                        R0,
                        initial_S,
                        initial_E,
                        initial_I,
                        initial_R,

                        initial_Sv,
                        initial_Ev,
                        initial_Iv,
                        initial_Rv,

                        population,
                        contact_matrix,
                        death_rates,
                        60
                    )





                    # # Baseline
                    w = np.zeros((problem.N + 1, num_age_groups))
                    S, E, I, R, Sv, Ev, Iv, Rv = simulate(problem, w)
                    dir_path = make_result_directory_for_simulation(state_id, contact_matrix_pair[0], R0, percentage_essential, "baseline_")
                    generate_all_plots(
                        dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False
                    )
                    #print(I)
                    cost_deaths1 = np.sum(R[-1, :] * problem.death_rates)



                    #print_heat_map(dir_path, w, S, E, I, R, V, contact_matrix_pair[0], percentage_essential, problem, False)

                    # Descent bang bang
                    w = learn_optimal_control_fixed_order(
                        problem, ordering, max_num_vaccines_per_day
                    )
                    S, E, I, R, Sv, Iv, Ev, Rv, = simulate(problem, w)
                    dir_path = make_result_directory_for_simulation(state_id, contact_matrix_pair[0], R0, percentage_essential, "descent_bangbang_")
                    generate_all_plots(
                        dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False
                    )

                    print_heat_map(dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False)
                    cost_deaths2 = np.sum(R[-1, :] * problem.death_rates)

                    # Descent bang bang with 5 E first

                    w = learn_optimal_control_fixed_order_5_E_First(
                        problem, ordering, max_num_vaccines_per_day
                    )

                    S, E, I, R, Sv, Iv, Ev, Rv = simulate(problem, w)
                    dir_path = make_result_directory_for_simulation(state_id, contact_matrix_pair[0], R0, percentage_essential, "descent_bangbang_5_Essentail_first_")
                    generate_all_plots(
                        dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False
                    )
                    print_heat_map(dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False)
                    cost_deaths3 = np.sum(R[-1, :] * problem.death_rates)

                    #Optimal control no init
                    S, E, I, R, Sv, Iv, Ev, Rv, w, u = solve_control_problem(problem, max_num_vaccines_per_day)
                    dir_path = make_result_directory_for_simulation(state_id, contact_matrix_pair[0], R0, percentage_essential, "opt_control_")
                    generate_all_plots(
                        dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False
                    )
                    print_heat_map(dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False)
                    cost_deaths4 = np.sum(R[-1, :] * problem.death_rates)

                    S, E, I, R, Sv, Iv, Ev, Rv = simulate(problem, w)
                    dir_path = make_result_directory_for_simulation(state_id, contact_matrix_pair[0], R0, percentage_essential, "simulated_opt_control_")
                    generate_all_plots(
                        dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False
                    )
                    print_heat_map(dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False)
                    cost_deaths5 = np.sum(R[-1, :] * problem.death_rates)

                    #Evenly spaced days
                    S, E, I, R, Sv, Iv, Ev, Rv = simulate(problem, w)
                    w = learn_optimal_control_fixed_day(
                        problem, ordering, max_num_vaccines_per_day)
                    dir_path = make_result_directory_for_simulation(state_id, contact_matrix_pair[0], R0, percentage_essential, "simulated_fixed_day_control_")
                    generate_all_plots(
                        dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False
                    )
                    print_heat_map(dir_path, w, S, E, I, R, Sv, Iv, Ev, Rv, contact_matrix_pair[0], percentage_essential, problem, False)

                    cost_deaths6 = np.sum(R[-1, :] * problem.death_rates)

                    writer.writerow([f"R0-{problem.R0} PE-{percentage_essential} beta-{contact_matrix_pair[0]} ", cost_deaths1, cost_deaths2,cost_deaths3,cost_deaths4,cost_deaths5, cost_deaths6 ])


if __name__ == "__main__":
    main()
