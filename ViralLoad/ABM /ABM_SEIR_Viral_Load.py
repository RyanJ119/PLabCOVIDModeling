import numpy as np
import random
import matplotlib.pyplot as plt

# Define model parameters
num_agents = 500  # Number of agents in the simulation
num_infected = 5  # Number of initially infected agents
infection_rate = 0.005  # Probability of transmission per contact
latent_period = 5  # Period from getting infected to becoming infectious
infectious_period = 14  # Duration of the infectious period in time steps
time_steps = 100  # Number of time steps in the simulation
immune_period = 7  # Number of days agent is immune from reinfection

# Empty list to append the average viral loads at each time step
viral_loads = []

# Viral load thresholds to determine when agents change compartments
thresh1 = 0.05
thresh2 = 0.9
thresh3 = 0.2


# Define agent class
class Agent:
    def __init__(self, state, viralload, age):
        self.state = state
        self.days_in_compartment = 0
        self.viralload = viralload
        self.immune_days = 0
        self.age = age
    def update_state(self, neighbors):
        if self.state == 'S':
            self.days_in_compartment += 1
            for neighbor in neighbors:
                if neighbor.state == 'I' and random.random() < infection_rate:
                    self.viralload += random.random() / 3
            if self.viralload > thresh1:
                self.state = 'E'
                self.days_in_compartment = 0
        elif self.state == 'E':
            self.days_in_compartment += 1
            self.viralload += random.random() / 3
            if self.days_in_compartment < latent_period and self.viralload > thresh2:
                self.state = 'I'
                self.days_in_compartment = 0
            elif self.days_in_compartment >= latent_period:
                self.state = 'R'
                self.days_in_compartment = 0

        elif self.state == 'I':
            self.days_in_compartment += 1
            self.viralload -= random.random() / 3
            self.viralload = max(self.viralload, 0)  # Prevent viral load from going below zero
            if self.viralload <= thresh3:
                self.state = 'R'
                self.days_in_compartment = 0

        elif self.state == 'R':
            self.days_in_compartment += 1
            self.viralload -= self.viralload
            # # ## Adds reinfectivity
            # if self.viralload <= 0:
            #     if self.immune_days >= immune_period:  # Check if the agent's immunity period is over
            #         self.state = 'S'
            #         self.days_in_compartment = 0
            #         self.immune_days = 0    # Reset the immune days counter
            #     else:
            #         self.immune_days += 1
    def get_state(self):
        return self.state
    def get_age(self):
        return self.age


# Define simulation function
def simulate():
    # Define age groups and probabilities
    age_groups = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-100']
    age_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # Normalize probabilities
    age_probs_normalized = [prob / sum(age_probs) for prob in age_probs]

    # Initialize agents
    agents = []
    for i in range(num_agents):
        if i < num_infected:
            state = 'I'
            viralload = (thresh2 + thresh3) / 2
        else:
            state = 'S'
            viralload = 0

        # Assign age based on age groups and probabilities
        age_group = np.random.choice(age_groups, p=age_probs_normalized)
        age_range = age_group.split('-')
        age = random.randint(int(age_range[0]), int(age_range[1]))

        agent = Agent(state, viralload, age)
        agents.append(agent)

    # Run simulation
    state_counts = []
    viral_load_data = [[] for _ in range(num_agents)]
    for t in range(time_steps):
        # Update agent states
        for agent in agents:
            neighbors = [neighbor for neighbor in agents if neighbor != agent]
            agent.update_state(neighbors)

        # # Interact n randomly chosen agents and add viral load
        # for _ in range(500):  # Choose 5 random interactions per time step
        #     random_agents = random.sample(agents, 2)
        #     agent1, agent2 = random_agents[0], random_agents[1]
        #
        #     if agent1.get_state() == 'S' and agent2.get_state() == 'I':
        #         susceptible_agent = agent1
        #         infected_agent = agent2
        #     elif agent1.get_state() == 'I' and agent2.get_state() == 'S':
        #         susceptible_agent = agent2
        #         infected_agent = agent1
        #     else:
        #         continue
        #
        #     susceptible_agent.viralload += infected_agent.viralload / 3

        # Record state counts
        s_count = sum([1 for agent in agents if agent.get_state() == 'S'])
        e_count = sum([1 for agent in agents if agent.get_state() == 'E'])
        i_count = sum([1 for agent in agents if agent.get_state() == 'I'])
        r_count = sum([1 for agent in agents if agent.get_state() == 'R'])
        state_counts.append([s_count, e_count, i_count, r_count])

        ## Calculate the average viral load for all agents
        avg_viral_loads = sum(agents.viralload for agents in agents if agents.get_state()) / len(agents)
        viral_loads.append(avg_viral_loads)

        # Append viral load data for each agent at the current time step
        for i, agent in enumerate(agents):
            viral_load_data[i].append(agent.viralload)


        # Print ages of all agents
    for i, agent in enumerate(agents):
        print(f"Agent {i + 1} age: {agent.get_age()}")

    # # Write viral load data to a file
    # with open('viral_load.csv', 'w') as file:
    #     for agent_loads in viral_load_data:
    #         file.write(','.join(str(load) for load in agent_loads) + '\n')

    return state_counts

# Run simulation and plot results
state_counts = simulate()
state_counts = np.array(state_counts)

s_counts = state_counts[:, 0]
e_counts = state_counts[:, 1]
i_counts = state_counts[:, 2]
r_counts = state_counts[:, 3]

## Plot SEIR dynamics for each state of agents over time
plt.figure(figsize=(10, 8))
plt.plot(s_counts, label='Susceptible')
plt.plot(e_counts, label='Exposed')
plt.plot(i_counts, label='Infected')
plt.plot(r_counts, label='Recovered')
plt.xlabel('Time steps')
plt.ylabel('Number of agents')
plt.title('Agent-based SEIR model simulation')
plt.legend()
plt.show()

## Collect time total steps in a vector
step_count = []
for steps in range(time_steps):
    step_count.append(steps)

# Plot the average viral loads over time
plt.figure(figsize=(10, 8))
plt.plot(step_count, viral_loads, label='Total Viral Load', color='purple')
plt.title('Average Viral Load Over Time')
plt.xlabel('Time Steps')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.ylabel('Viral Load')
plt.legend()
plt.show()


