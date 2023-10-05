
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('data/GeneralEsportData.csv')
Df.plot(x='releaseDate', y='TotalTournaments', kind = 'scatter')
plt.show()