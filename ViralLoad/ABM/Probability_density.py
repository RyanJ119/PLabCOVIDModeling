import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
df = pd.read_csv('viral_load.csv')

# Generate a grid of X, Y, and Z values
X = np.arange(df.shape[1])  # Time steps
Y = np.arange(df.shape[0])  # Persons
X, Y = np.meshgrid(X, Y)

# Extract the viral load data from the DataFrame
Z = df.values

# Normalize the viral load data between 0 and 1
Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the heat map with normalized values
ax.plot_surface(X, Y, Z_normalized, cmap='viridis')

# Set labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Agents')
ax.set_zlabel('Viral Load Probability')
ax.set_title('Probability Density of Viral Load')

# Display the plot
plt.show()