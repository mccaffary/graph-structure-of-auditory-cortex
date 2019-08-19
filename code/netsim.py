import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import itertools
from itertools import combinations

# script for generating synethic networks (with known organisational principles) on which the graphsage-based algorithm can be tested

#generation of node class

class node:
    
    def __init__(self,n_ftrs,n_neighbours):
        
        """
        initialiser
        
        Arguments:
        =========================
        n_ftrs:        ftrs of neighbours
        n_neighbours:  number of neighbours
        
        """
        
        self.n_ftrs = int(n_ftrs)
        self.n_neighbours = int(n_neighbours)
        self.ftrs = np.random.normal(size=self.n_ftrs)
        
        self.get_neighbourhood()
        self.mean_aggregate()
        
        
    def get_neighbourhood(self):
        
        """
        generates neighbourhood for each given node
       
        """

        mu, var = np.random.normal(loc=0,scale=4,size=2)
        var = np.abs(var)
        self.neigh_feats = np.random.normal(loc=mu,scale=var,size=(self.n_neighbours,self.n_ftrs))
        #self.neigh_feats = np.random.normal(loc=mu,scale=var,size=(np.random.uniform(int(self.n_neighbours*.8),int(self.n_neighbours*1.2),self.n_ftrs)))
        #conn_str = lambda x,y
        
    def mean_aggregate(self):
        self.agg = np.mean(self.neigh_feats, axis=0)

#generation of conn_mtx of give node from ftrs of neighbours

def connectivity_matrix_neigh_feats(n_ftrs,n_neighbours,n_testNs):
    
    """
    Generates conn_mtx of given node from ftrs of neighbours
    
    Arguments:
    ========================
    n_ftrs:        int
                   ftrs of neighbours
    n_neighbours:  int
                   number of neighbours
    n_testNs:      int
                   number of test nodes (from which to generate conn_mtx)
    
    """
    
    node_set = []
    for i in range(n_testNs):
        node_set.append(node(n_ftrs,n_neighbours))

    W = np.random.normal(size=n_ftrs*4)
    #nonLin = lambda x: np.tanh(x)
    conn_mtx_n = np.zeros([n_testNs,n_testNs])

    for i1,pair1 in enumerate(node_set):
        for i2,pair2 in enumerate(node_set):
            conn_mtx_n[i1,i2] = (W.dot(np.concatenate([pair1.ftrs,pair1.agg,pair2.ftrs,pair2.agg])))
    return conn_mtx_n, W, node_set
    
