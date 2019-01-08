from VisualizeTrees import *
import time
import pyslim
from matplotlib import colors
from matplotlib import pyplot as plt

ts = msprime.simulate(500,recombination_rate=10.0)
ts = pyslim.load("/Users/jaredgalloway/Desktop/FIXED7.trees")

data = VisualizeNodes(treeSequence = ts,rescaled_time=False,RowsInImage=1000,ColumnsInImage=1000)

rgb = colors.hsv_to_rgb(data)
plt.imshow(rgb)
plt.show()
