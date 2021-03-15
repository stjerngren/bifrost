import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.6  # previous pdf hatch linewidth
mpl.rcParams['hatch.linewidth'] = 6.0  # previous svg hatch linewidth
 
nice_fonts = {
    "text.usetex": False,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 30,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 30,
    "xtick.labelsize": 20,
    "ytick.labelsize": 30,
}
 
mpl.rcParams.update(nice_fonts)


labels = [
    'Conv 1',
    'Conv 2',
    'Conv 3',
    'Conv 4',
    'Conv 5',
    'Average'
    ]
basic = [ 
    280549888,
	895683264,
	449031552,
	598837504,
	399182080, 
]



basic.append(sum(basic)/len(basic))
bifrost = [ 3679680,19044096,8843136,12133632,12133120]
bifrost.append(sum(bifrost)/len(bifrost))

print((1-(bifrost[1]/basic[-1]))*100)
test = [1 / (i/j) for i, j in zip(bifrost, basic)]
#print(test)
#
#bifrost_norm = [float(i)/sum(bifrost) for i in bifrost]
#
#
#plt.figure(figsize=((1+(5**1/2))/2*10,10))
#plt.grid( axis='y')
#y_pos = np.arange(len(labels))
#performance = [1 / (i/j) for i, j in zip(bifrost, basic)]
#performance.append(sum(performance)/len(performance))
#plt.bar(y_pos, performance, width = 0.5, align='center', alpha=0.8)
#plt.xticks(y_pos, labels)
#plt.ylabel('Speedup')
##plt.axes().set_aspect(0.61803398875)
#plt.savefig("alexnet_maeri.pdf", bbox_inches='tight')
#plt.show()

# CONV 1,  CONV 2,  CONV 3,  CONV 4,  CONV 5 
# 3679680,23956800,9185664,13668992,9120000 stonne paper config (from mRNA)
# 3679680,19044096,8843136,12133632,12133120 autoTVM
mrna = [3679680,15264384,7930368,10550400,7041408,] # mRNA performance mode



label = ["CONV1","CONV2","CONV3","CONV4","CONV5","FC1","FC2","FC3"]
sparse_0 = [ 		1164224,
		3671808,
		1887360,
		2515712,
		1706752,]
sparse_0.append(sum(sparse_0)/len(sparse_0))

sparse_50 = [ 1036768, 2056417, 1042115, 1211249, 808686, ]
sparse_50.append(sum(sparse_50)/len(sparse_50))

basic = " & ".join(["{:.1e}".format(x) for x in basic])
bifrost = " & ".join(["{:.1e}".format(x) for x in bifrost])
sparse_50 = " & ".join(["{:.1e}".format(x) for x in sparse_50])
sparse_0 = " & ".join(["{:.1e}".format(x) for x in sparse_0])

print(basic)
print(bifrost)
print(sparse_0)
print(sparse_50)
#x = np.arange(len(label))  # the label locations
#width = 0.35  # the width of the bars
#
#plt.figure(figsize=((1+(5**1/2))/2*10,10))
#plt.bar(x - width/2, sparse_0, width, label='SIGMA 0% Sparsity')
#plt.bar(x + width/2, sparse_50, width, label='SIGMA 50% Sparsity')
## Add some text for labels, title and custom x-axis tick labels, etc.
#plt.xticks(x, label)
#plt.ylabel('Clock Cycles')
#plt.legend()
#plt.show()



#(1+(5**1/2))/2




## libraries
#import numpy as np
#import matplotlib.pyplot as plt
# 
## set width of bar
#barWidth = 0.25
# 
## set height of bar
# 
## Set position of bar on X axis
#r1 = np.arange(len(basic))
#r2 = [x + barWidth for x in r1]
#r3 = [x + barWidth for x in r2]
# 
## Make the plot
#plt.bar(r1, basic, width=barWidth, label='Basic Tile Config')
#plt.bar(r2, bifrost,  width=barWidth, label='Bifrost Tile Config')
#plt.bar(r3, mrna, width=barWidth, label='mRNA tile config')
# 
## Add xticks on the middle of the group bars
#plt.xlabel('group', fontweight='bold')
#plt.xticks([r + barWidth for r in range(len(basic))], ['A', 'B', 'C', 'D', 'E'])
# 
## Create legend & Show graphic
#plt.legend()
#plt.show()
