# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 
font = {'family': 'arial',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }



df = pd.DataFrame({
'group': ['A'],
'Model 1': [92.98],
'Model 2': [73.73],
'Model 3': [89.68],
'Model 4': [92.98],
'Model 5': [89.68],
'Model 6': [94.85],
'Model 7': [94.85]
}) 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
values
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
#title="F1 scores of The Proposed Models"
title =""
ax = plt.subplot(111, polar=True)
ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center', fontdict=font)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories, color='black', size=10)
 
# Draw ylabels
ax.set_rlabel_position(5)
plt.yticks([70,80,90,100], ["70","80","90","100"], color="red", size=10)
plt.ylim(0,100)
 
# Plot data
ax.plot(angles, values, linewidth=2, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

# Show the graph
plt.show()