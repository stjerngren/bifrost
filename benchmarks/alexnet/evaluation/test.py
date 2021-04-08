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
#basic = [ 
#    280549888,
#	895683264,
#	449031552,
#	598837504,
#	399182080, 
#]
#
#
#
#bifrost = [ 3679680,19044096,8843136,12133632,12133120]
#
#
#plt.figure(figsize=((1+(5**1/2))/2*10,10))
#plt.grid( axis='y')
#y_pos = np.arange(len(labels))
#performance = [1 / (i/j) for i, j in zip(bifrost, basic)]
#print(performance)
#performance.append(sum(performance)/len(performance))
#
#plt.bar(y_pos, performance, width = 0.5, align='center', alpha=0.8)
#plt.xticks(y_pos, labels)
#plt.ylabel('Speedup')
##plt.axes().set_aspect(0.61803398875)
##p#lt.savefig("alexnet_maeri.pdf", bbox_inches='tight')
#plt.show()

# CONV 1,  CONV 2,  CONV 3,  CONV 4,  CONV 5 
# 3679680,23956800,9185664,13668992,9120000 stonne paper config (from mRNA)
# 3679680,19044096,8843136,12133632,12133120 autoTVM

#
#autoTVM = [3679680,19044096,8843136,12133632,12133120]
#autoTVM.append(sum(autoTVM)/len(autoTVM))
#
#mrna = [3679680,15264384,7930368,10550400,7041408,] # mRNA performance mode
#mrna.append(sum(mrna)/len(mrna))
#x = np.arange(len(labels))  # the label locations
#width = 0.35  # the width of the bars
#
#plt.figure(figsize=((1+(5**1/2))/2*10,10))
#plt.bar(x - width/2, autoTVM, width, label="Bifrost Tile Config")
#plt.bar(x + width/2, mrna, width, label='mRNA Tile Config')
## Add some text for labels, title and custom x-axis tick labels, etc.
#plt.xticks(x, labels)
#plt.ylabel('Clock Cycles')
#
#plt.legend()
#plt.savefig("alexnet_maeri_vs_mrna.pdf", bbox_inches='tight')
#
#plt.show()


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



label = ["CONV1","CONV2","CONV3","CONV4","CONV5","FC1","FC2","FC3"]
sparse_0 = [ 				1164224,
		3671808,
		1887360,
		2515712,
		1706752,
		4485120,
		2019328,
		493000,]

sparse_50 = [		1036768,
		2056417,
		1042115,
		1211249,
		808686,
		4485120,
		2019328,
		493000]


#sparse_50 = " & ".join(["{:.1e}".format(x) for x in sparse_50])
#sparse_0 = " & ".join(["{:.1e}".format(x) for x in sparse_0])

#
#print(sparse_0)
#print(sparse_50)
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
#plt.savefig("alexnet_sigma.pdf", bbox_inches='tight')
#plt.show()
#

#(1+(5**1/2))/2





#labels = ["CONV1","CONV2","CONV3","CONV4","CONV5"]
#
## libraries
#import numpy as np
#import matplotlib.pyplot as plt
# 
## set width of bar
#barWidth = 0.3
#
#plt.figure(figsize=((1+(5**1/2))/2*10,10))
## set height of bar
# 
#basic = [ 
#    280549888,
#	895683264,
#	449031552,
#	598837504,
#	399182080, 
#
#]
#bifrost = [ 3679680,19044096,8843136,12133632,12133120, ]
#mrna = [3679680,15264384,7930368,10550400,7041408]
#
#x = np.arange(len(labels))
## Set position of bar on X axis
#
# 
#ax = plt.subplot(111)
#ax.bar(x-0.2, basic, width=0.2, align='center', label='Basic Tile Config')
#ax.bar(x, bifrost, width=0.2,  align='center', label='Bifrost Tile Config')
#ax.bar(x+0.2, mrna, width=0.2,  align='center', label='mRNA Tile Config')
#plt.xticks(x, labels)
#plt.ylabel('Clock Cycles')
#plt.legend()
#plt.savefig("alexnet_fc.pdf", bbox_inches='tight')
#plt.show()


#plt.figure(figsize=((1+(5**1/2))/2*10,10))
#labels = ["FC1","FC2","FC3", "Average"]
#x = np.arange(len(labels))
#y = [188747776,
#		83890176,
#		20481000]
#z = [		17006592,
#		7569408,
#		1848000,]
#k = [		4242432,
#		2624512,
#		467100,]
#z.append(sum(z)/len(z))
#k.append(sum(k)/len(k))
#
#x = np.arange(len(labels))  # the label locations
#width = 0.35  # the width of the bars
#
#plt.figure(figsize=((1+(5**1/2))/2*10,10))
#plt.bar(x - width/2, z, width, label="Bifrost Mapping")
#plt.bar(x + width/2, k, width, label='mRNA Mapping')
## Add some text for labels, title and custom x-axis tick labels, etc.
#plt.xticks(x, labels)
#plt.ylabel('Clock Cycles')
#
#plt.legend()
#
#plt.savefig("alexnet_fc_maeri_mrna.pdf", bbox_inches='tight')
#plt.show()

#"""Fully connected"""
#labels = [
#    'FC 1',
#    'FC 2',
#    'FC 3',
#    'Average'
#    ]
#basic = [188747776,83890176,20481000,]
#bifrost = [ 17006592,7569408,1848000]
#
#
#plt.figure(figsize=((1+(5**1/2))/2*10,10))
#plt.grid( axis='y')
#y_pos = np.arange(len(labels))
#performance = [1 / (i/j) for i, j in zip(bifrost, basic)]
#print(performance)
#performance.append(sum(performance)/len(performance))
#
#plt.bar(y_pos, performance, width = 0.5, align='center', alpha=0.8)
#plt.xticks(y_pos, labels)
#plt.ylabel('Speedup')
##plt.axes().set_aspect(0.61803398875)
#plt.savefig("alexnet_maeri_fc_sppedup.pdf", bbox_inches='tight')
#plt.show()


sparse_0 = [ 
    1164224,
	3671808,
	1887360,
	2515712,
	1706752,
	4485120,
	2019328,
	493000,]

sparse_50 = [
    1036768,
	2056417,
	1042115,
	1211249,
	808686,
	4485120,
	2019328,
	493000]


basic_conv = [ 
    280549888,
	895683264,
	449031552,
	598837504,
	399182080, 

]
bifrost_conv = [ 3679680,19044096,8843136,12133632,12133120, ]
mrna_conv = [3679680,15264384,7930368,10550400,7041408]
basic_conv.append(sum(basic_conv)/len(basic_conv))
bifrost_conv.append(sum(bifrost_conv)/len(bifrost_conv))
mrna_conv.append(sum(mrna_conv)/len(mrna_conv))


basic_fc = [188747776,
		83890176,
		20481000]
bifrost_fc_mapping = [		17006592,
		7569408,
		1848000,]
mrna_fc_mapping = [		4242432,
		2624512,
		467100,]
basic_fc.append(sum(basic_fc)/len(basic_fc))
bifrost_fc_mapping.append(sum(bifrost_fc_mapping)/len(bifrost_fc_mapping))
mrna_fc_mapping.append(sum(mrna_fc_mapping)/len(mrna_fc_mapping))


print(" & ".join(["{:.2e}".format(x) for x in sparse_0]))
print(" & ".join(["{:.2e}".format(x) for x in sparse_50]))

print(" & ".join(["{:.2e}".format(x) for x in basic_conv]))
print(" & ".join(["{:.2e}".format(x) for x in bifrost_conv]))
print(" & ".join(["{:.2e}".format(x) for x in mrna_conv]))

print(" & ".join(["{:.2e}".format(x) for x in basic_fc]))
print(" & ".join(["{:.2e}".format(x) for x in bifrost_fc_mapping]))
print(" & ".join(["{:.2e}".format(x) for x in mrna_fc_mapping]))
