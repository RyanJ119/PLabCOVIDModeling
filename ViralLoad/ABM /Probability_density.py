import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('viral_load.csv', header = None)

def my_kde(data, width=.35, gridsize=100, normalized=True, bounds=None):
    """
    Compute the gaussian KDE from the given sample.

    Args:
        data (array or list): sample of values
        width (float): width of the normal functions
        gridsize (int): number of grid points on which the kde is computed
        normalized (bool): if True the KDE is normalized (default)
        bounds (tuple): min and max value of the kde

    Returns:
        The grid and the KDE
    """
    # boundaries
    if bounds:
        xmin, xmax = bounds
    else:
        xmin = min(data) - 3 * width
        xmax = max(data) + 3 * width

    # grid points
    x = np.linspace(xmin, xmax, gridsize)

    # compute kde
    kde = np.zeros(gridsize)
    for val in data:
        kde += norm.pdf(x, loc=val, scale=width)

    # normalized the KDE
    if normalized:
        kde /= sp.integrate.simps(kde, x)

    return x, kde

x, kde = my_kde(df[15], gridsize=100)
''
x = np.ones( (200,100) )
kde =  np.ones( (200,100) )
for i in range(1,100):

    x[:,i], kde[:,i] = my_kde(df[i], gridsize=200)


ax = sns.distplot(df[15], hist=False, axlabel="Something ?",
                  kde_kws=dict(gridsize=200, bw=1, label="seaborn"))

ax.plot(x, kde, "o", label="my_kde")
plt.legend();


X = np.arange(df.shape[1])  # Time steps
#X =np.linspace(0,100,100)  # Time steps

Y = x # Persons
Z = kde
#X, Y = np.meshgrid(X, Y)


#Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the heat map with normalized values
ax.plot_surface(X, Y, Z, cmap='viridis')

# Set labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Viral Load')
ax.set_zlabel('Probability Density')
ax.set_title('Probability Density of Viral Load')

# Display the plot
plt.show()
