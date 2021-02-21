import matplotlib
import matplotlib.pyplot as plt
import numpy as np


import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.6  # previous pdf hatch linewidth
mpl.rcParams['hatch.linewidth'] = 6.0  # previous svg hatch linewidth
 
nice_fonts = {
    "text.usetex": False,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 40,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
}
 
mpl.rcParams.update(nice_fonts)


labels = [
    'Conv 1',
    'Conv 2',
    'Conv 3',
    'Conv 4',
    'Conv 5'
    ]
women_means = [ 9296576, 19110336, 8975616, 12307968, 12307968 ]
men_means = [ 9296576, 24015936, 9315968, 13833984, 9231872 ]

#3292160, 16941312, 7863168, 10784000, 10783488 
print(sum(women_means))
print(sum(men_means))
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='mRNA tile config')
rects2 = ax.bar(x + width/2, women_means, width, label='Bifrost tile config')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Cycles')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()