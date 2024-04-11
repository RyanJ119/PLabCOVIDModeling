import numpy as np

from utils import read_data_from_csv

# State used for the simulations
state_id = "NJ"

# COVID 19 data
data = read_data_from_csv("../data/covid19_data_v1.csv")
# Interaction matrice
matrices = [(0.75,  "../data/InteractionMatrix_beta_0_75.csv")]
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
cost_lockdown = 70 # $ per day per person
num_age_groups = len(death_rates[0])
time_horizon = 180 # days
R0s = [1.7]
percentages_essential = [0]
