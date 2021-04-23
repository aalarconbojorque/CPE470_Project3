# -----------------------------------------------------------------------------
# FILE NAME:         proj3.py
# USAGE:             python3 proj3.py
# NOTES:             Requires NumPy installation
#                    Requires Python3
#
# MODIFICATION HISTORY:
# Author             Date           Modification(s)
# ----------------   -----------    ---------------
# Andy Alarcon       04-01-2021     1.0 ... Setup dev environment, imported NumPy
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random


def main():
    r = 1 #Set Communication Range
    num_nodes = 10 #Randomly generated nodes
    delta_t_update = 0.008 
    n = 2 #Number of dimensions aaaddddddd dsssdd ffff
    nodes = np.random.randn(num_nodes, n)

    #Add measurment for each node yi = theta_t + v_i
    nodes_va = (50 * np.ones((num_nodes, 1))) + (1 * np.random.randn(num_nodes, 1))
    nodes_va0 = nodes_va #Save inital measurments

    #Find the neighbors of the nodes
    Neigbors = FindNeighbors(nodes, r , n, delta_t_update)


    G = nx.Graph()

    #Add edges
    for i, item in enumerate(Neigbors, start=0):
        G.addEdge(Neigbors[i])

    for i, item in enumerate(nodes, start=0):
        print("Node ", i , " X :" ,nodes[i][0], " Y :", nodes[i][1])

    nod = {node: node for node in nodes}

    fig, ax = plt.subplots()
    nx.draw(G, nod=nod, node_color='k', ax=ax)
    nx.draw(G, nod=nod, node_size=1500, ax=ax)
    nx.draw_networkx_labels(G, nod=nod) 
    plt.axis("on")
    ax.set_xlim(0, 11)
    ax.set_ylim(0,11)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()

   # G.visualize()

    print("Data for target, robot and errhhor written")
    
# class GraphVisualization:
   
#     def __init__(self):        
#         self.points = []
#         self.edges = []
          
#     # addEdge function inputs the vertices of an
#     # edge and appends it to the visual list
#     def addEdge(self, pair):
#         G.add_edge(pair)
          
#     # In visualize function G is an object of
#     # class Graph given by networkx G.add_edges_from(visual)
#     # creates a graph with a given list
#     # nx.draw_networkx(G) - plots the graph
#     # plt.show() - displays the graph
#     def visualize(self):
#         G = nx.Graph()
#         G.add_edges_from(self.visual)
#         nx.draw_networkx(G)
#         plt.show()
    

# ----------------------------------------------------------------------------
# FUNCTION NAME:     CompMagSqr()
# PURPOSE:           Computes the magnitude squared
# -----------------------------------------------------------------------------
def FindNeighbors(nodes, r, n, delta_t):

    Neigbor_array = []

    #Loop through each node and compare it to all other nodes
    for i, item in enumerate(nodes, start=0): 
        print("Looking at point " , i)
        for j, item in enumerate(nodes, start=0):

            #Calculate distance between node i and j
            neighborDistance = np.sqrt(np.square(nodes[j][0] - nodes[i][0]) + np.square(nodes[j][1] - nodes[i][1]))
            #If within range they are neighbors
            if(neighborDistance <= r and neighborDistance != 0):
                Neigbor_array.append(np.array([i,j]))
                #print("Node " , i , "->" , j)
                #print("Distance :" , np.sqrt(np.square(nodes[j][0] - nodes[i][0]) + np.square(nodes[j][1] - nodes[i][1])))

    return Neigbor_array

    
if __name__ == "__main__":
    main()