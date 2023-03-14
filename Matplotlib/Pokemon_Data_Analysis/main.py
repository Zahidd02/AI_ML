import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import random

# show number of Pokémon with similar powers
df = pd.read_csv('pokemon_alopez247.csv')
type_1 = df.iloc[:, 2].value_counts(sort=False)
type_1_names = type_1.index
type_1_count = type_1.values
type_1_grps = np.arange(len(type_1))

# (optional) changes color for each bar in the chart
clrs = np.linspace(0, 1, 18)
random.shuffle(clrs)
colors = []
for i in range(0, 72, 4):
    idx = np.arange(0, 18, 1)
    random.shuffle(idx)
    r = clrs[idx[0]]
    g = clrs[idx[1]]
    b = clrs[idx[2]]
    a = clrs[idx[3]]
    colors.append([r, g, b, a])

bar_width = 0.5
bar_graph = plt.bar(type_1_grps, type_1_count, bar_width, alpha=0.5, color=colors,
                    label='Pokemon count respective to their Type_1')
plt.legend(bar_graph, type_1_names)
plt.xticks(type_1_grps, type_1_names, rotation='vertical')
plt.grid()
plt.ylim(0, 130)
# plt.show()
plt.close()

# compare the Attack, Defence, Speed and HP of top 4 major groups
df_pie = df[['Type_1', 'Attack', 'Defense', 'Speed', 'HP']].copy()
frequent_grp = df_pie['Type_1'].value_counts().nlargest(4)
df_pie = df_pie[df_pie['Type_1'].str.contains(r'(Water|Normal|Grass|Bug)')]
type_1_names = frequent_grp.index
df_grp = df_pie.groupby('Type_1').mean()
# print(df_grp)

names = df_grp.columns
colors = ['gold', 'lightcoral', 'yellowgreen', 'lightskyblue']
explode = (0, 0, 0, .1)  # takes out only the 4th slice

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
ax = [ax1, ax2, ax3, ax4]

# plots the graph in single window
for i in range(0, 4):
    percent = df_grp.iloc[i, :]
    ax[i].pie(percent, explode=explode, labels=names, colors=colors, autopct='%.2f%%', shadow=False, startangle=90)
    ax[i].set_aspect('equal')
    ax[i].set_title(type_1_names[i])
# plt.show()
plt.close(fig)

# trying to find any relation between the Pokémon catch rate and total power
total_power = df.iloc[:, 4]
catch_rate = df.iloc[:, 21]

fig, ax = plt.subplots()
p = ax.scatter(catch_rate, total_power, c='g')
ax.set_xlabel('Catch Rate')
ax.set_ylabel('Total Power')
ax.set_title('Pokemon Catch Rate vs their Power')
ax.grid()
plt.legend([p], ['Pokemons'])

trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

rect = patches.Rectangle((0, 300), width=0, height=5, transform=trans, color='red', alpha=0.4)
ax.add_patch(rect)
plt.show()
