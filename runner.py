from VisualizeTrees import *
import time
#import pyslim
from matplotlib import colors
from matplotlib import pyplot as plt

#simulate a new tree sequence from coalescent
ts = msprime.simulate(500,recombination_rate=10.0)

#or load one from pyslim/msprime
#ts = pyslim.load("/Path/to/example.trees")

#Get the 3-D tensor containing the hsv values across a given number of rows and columns
data = VisualizeNodes(treeSequence = ts,rescaled_time=False,RowsInImage=1000,ColumnsInImage=1000)

#Convert to an image
rgb = colors.hsv_to_rgb(data)
plt.imshow(rgb)
plt.show()
