import msprime
import itertools
import numpy as np
from weighted_trees import *

def VisualizeNodes(treeSequence,rescaled_time=True,RowsInImage=1000,ColumnsInImage=1000):

    """
    A function designed to take in a msprime (or pyslim) treeSequence object,
    and convert it into a 3-D tensor of shape (Rows,Columns,3). 

    you can then use:
    ```
    rgb = matplotlib.colors.hsv_to_rgb(data)
    matplotlib.imshow(rgb)
    matplotlib.show()```

    to plot the visualization. 
    
    Currently, we give all of the treeSequence samples
    a unique hue uniformily distributed around the 
    Hue wheel. The Common ancestor of any set of samples 
    is then the average vector of all of its children. 

    Saturation is then determined by the number of tracked samples 
    in the subtree rooted at any one node.
    
    Lightness is determined by the branch length of 
    any one node.

    return: np.array

    """

    treeSequence.simplify()    
    Ne = treeSequence.num_samples

    #color each sample evenly around the hue circle
    initial_hues = np.linspace(0,2*np.pi,Ne,endpoint=False)
    
    #The function for that will compute mean hue for the weight[0] 
    #of the parent node given its two children's hues.
    def angle_mean(a):
        # gives the angle of the average vector in the unit circle
        # https://en.wikipedia.org/wiki/Mean_of_circular_quantities
        return np.math.atan2(sum(np.sin(a)), sum(np.cos(a)))

    wt = weighted_trees(treeSequence, [initial_hues], angle_mean)

    #initialize empty tensor 
    data = np.zeros((RowsInImage,ColumnsInImage,3), dtype=np.float)
    #find oldest node to scale the trees by
    oldest_node = max([n.time for n in treeSequence.nodes()])
    if(rescaled_time):
        oldest_node = Ne - (1/(1/Ne + oldest_node))

    #get the correct column index interval for respective trees along the genome
    bp = (np.array([b for b in treeSequence.breakpoints()])*(ColumnsInImage/treeSequence.sequence_length)).astype(np.int)
    
    for i,t in enumerate(wt):

        nn = len([n for n in t.nodes()])

        node_times = np.repeat(0,nn)
        node_bl = np.repeat(0,nn)
        node_tsam = np.repeat(0,nn)
        
        for c,u in enumerate(t.nodes()):
            node_times[c] = treeSequence.node(u).time
            node_tsam[c] = t.num_tracked_samples(u)
            if t.parent(u) != msprime.NULL_NODE:
                node_bl[c] = t.branch_length(u)
            else:
                node_bl[c] = 0.0
        
        node_times = np.array([treeSequence.node(u).time for u in t.nodes()])
        if(rescaled_time):
            node_times = Ne - (1/(1/Ne + node_times))

        #Get the correct row index for respective node times in each sparse tree
        node_rows = (node_times * (RowsInImage / oldest_node)).astype(np.int)
        node_hues = (np.array([w[0] for w in t.node_weights()])*((360/(2*np.pi)))%360).astype(np.int)
        node_tsam = node_tsam/Ne
        node_bl = 1 - (node_bl/oldest_node)
        colorStrings = [[h/360,t,b] for h,t,b in zip(node_hues,node_tsam,node_bl)]
        
        for l in range(len(node_rows)):
            data[((RowsInImage)-node_rows[l])-1][bp[i]:bp[i+1]] = colorStrings[l]         
        
    return data


def weighted_trees(ts, sample_weight_list, node_fun=sum):
    '''
    Here ``sample_weight_list`` is a list of lists of weights, each of the same
    length as the samples in the tree sequence ``ts``. This returns an iterator
    over the trees in ``ts`` that is identical to ``ts.trees()`` except that
    each tree ``t`` has the additional method `t.node_weights()` which returns
    an iterator over the "weights" for each node in the tree, in the same order
    as ``t.nodes()``.

    Each node has one weight, computed separately for each set of weights in
    ``sample_weight_list``. Each such weight is defined for a particular list
    of ``sample_weights`` recursively:

    1. First define ``all_weights[ts.samples()[j]] = sample_weights[j]``
        and ``all_weights[k] = 0`` otherwise.
    2. The weight for a node ``j`` with children ``u1, u2, ..., un`` is
        ``node_fun([all_weights[j], weight[u1], ..., weight[un]])``.

    For instance, if ``sample_weights`` is a vector of all ``1``s, and
    ``node_fun`` is ``sum``, then the weight for each node in each tree
    is the number of samples below it, equivalent to ``t.num_samples(j)``.

    To do this, we need to only recurse upwards from the parent of each
    added or removed edge, updating the weights.
    '''
    samples = ts.samples()
    num_weights = len(sample_weight_list)
    # make sure the provided initial weights lists match the number of samples
    for swl in sample_weight_list:
        assert(len(swl) == len(samples))    

    # initialize the weights
    base_X = [[0.0 for _ in range(num_weights)] for _ in range(ts.num_nodes)]
    X = [[0.0 for _ in range(num_weights)] for _ in range(ts.num_nodes)]
    #print(samples)
    for j, u in enumerate(samples):
        for k in range(num_weights):
            X[u][k] = sample_weight_list[k][j]
            base_X[u][k] = sample_weight_list[k][j]


    for t, (interval, records_out, records_in) in zip(ts.trees(tracked_samples=ts.samples()), ts.edge_diffs()):
        for edge in itertools.chain(records_out, records_in):
            u = edge.parent
            while u != msprime.NULL_NODE:
                for k in range(num_weights):
                    U = None
                    if(t.is_sample(u)):
                        U = [base_X[u][k]] + [X[u][k] for u in t.children(u)]               
                    else:
                        U = [X[u][k] for u in t.children(u)] 
                    X[u][k] = node_fun(U) 
                u = t.parent(u)

        def the_node_weights(self):
            for u in self.nodes():
                yield X[u]

        # magic that uses "descriptor protocol"
        t.node_weights = the_node_weights.__get__(t, msprime.trees.SparseTree)
        yield t



