mport matplotlib.pyplot as plt
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