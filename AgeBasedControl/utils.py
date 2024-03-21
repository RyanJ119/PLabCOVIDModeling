import numpy as np
import pandas as pd
import os


class Problem:
    def __init__(
        self,
        _time_horizon,
        _R0,
        _init_S,
        _init_E,
        _init_I,
        _init_R,
        
        
        
        _population,
        _contact_matrix,
        _death_rates,
        _cost_lockdown,
        _N=None,
        _delta=1 / 4.0,
        _gamma=1 / 5.0,
    ):
        self.N = _N
        if _N is None:
            self.N = _time_horizon
        self.time_horizon = _time_horizon
        self.R0 = _R0
        self.initial_S = _init_S
        self.initial_E = _init_E
        self.initial_I = _init_I
        self.initial_R = _init_R
       # self.initial_V = _init_V
        
        
        
        self.population = _population
        self.contact_matrix = _contact_matrix
        self.death_rates = _death_rates
        self.cost_lockdown = _cost_lockdown
        self.delta = _delta
        self.gamma = _gamma
        self.num_age_groups = _population.shape[1]


def read_data_from_csv(path):
    df = pd.read_csv(path, index_col="state")
    data = {}
    for idx, state in df.iterrows():
        data[idx] = {
            "R0": state["R0"],
            "total_vaccines_per_day": state["total_vaccines_per_day"],
            "population": np.array(
                [
                    [
                        state["population0"],
                        state["population1"],
                        state["population2"],
                        state["population3"],
                        state["population4"],
                        state["population5"],
                    ]
                ]
            ),
            "initial_S": np.array(
                [
                    [
                        state["initialS0"],
                        state["initialS1"],
                        state["initialS2"],
                        state["initialS3"],
                        state["initialS4"],
                        state["initialS5"],
                    ]
                ]
            ),
            "initial_E": np.array(
                [
                    [
                        state["initialE0"],
                        state["initialE1"],
                        state["initialE2"],
                        state["initialE3"],
                        state["initialE4"],
                        state["initialE5"],
                    ]
                ]
            ),
            "initial_I": np.array(
                [
                    [
                        state["initialI0"],
                        state["initialI1"],
                        state["initialI2"],
                        state["initialI3"],
                        state["initialI4"],
                        state["initialI5"],
                    ]
                ]
            ),
            "initial_R": np.array(
                [
                    [
                        state["initialR0"],
                        state["initialR1"],
                        state["initialR2"],
                        state["initialR3"],
                        state["initialR4"],
                        state["initialR5"],
                    ]
                ]
            ),
        }
    return data


def transform_to_have_essential_workers(data, percentage_essential=0.44):
    percentage_people_seventy = 0.35
    data6 = max(.000001, data[0, 5] * percentage_people_seventy)
    data5 = data[0, 5] * (1 - percentage_people_seventy)
    return np.array(
        [
            [
                data[0, 0],
                data[0, 1],
                data[0, 2] * (1 - percentage_essential),
                data[0, 2] * percentage_essential,
                data[0, 3] * (1 - percentage_essential),
                data[0, 3] * percentage_essential,
                data[0, 4] * (1 - percentage_essential),
                data[0, 4] * percentage_essential,
                data5 * (1 - percentage_essential),
                data5 * percentage_essential,
                data6,
            ]
        ]
    )


def make_result_directory_for_simulation(state_id, beta, R0, percentage_essential, prefix=""):
    results_path = os.path.join("../results/", f"{prefix}{state_id}_beta{beta}_R0{R0}_PE{percentage_essential}")
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass
    return results_path
