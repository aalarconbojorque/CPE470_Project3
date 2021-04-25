# -----------------------------------------------------------------------------
# FILE NAME:         proj3.py
# USAGE:             python3 proj3.py
# NOTES:             Requires matplotlib and numpy installation
#                    Requires networkx installation
#                    Requires Python3
#
# MODIFICATION HISTORY:
# Author             Date           Modification(s)
# ----------------   -----------    ---------------
# Andy Alarcon       04-23-2021     1.0 ... Setup dev environment, imported NumPy
# Andy Alarcon       04-25-2021     1.1 ... imported matpotlib and networkx, created graph display
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random

class GraphNode:
    def __init__(self, index, position, neighbors):
        self.index = index
        self.position = position
        self.neighbors = neighbors

    def FindyourNeighbors(self, nodesObjects, r):
      
      for j, item in enumerate(nodesObjects, start=0):
            # Calculate distance between self and j
            neighborDistance = np.sqrt(np.square(nodesObjects[j].position[0] - self.position[0]) + np.square(nodesObjects[j].position[1] - self.position[1]))
            # If within range they are neighbors
            if(neighborDistance <= r and neighborDistance != 0):
                self.neighbors.append(j)
               # print("Node ", self.index, "->", j)
                

def main():
    r = 1  # Set Communication Range
    num_nodes = 10  # Randomly generated nodes
    delta_t_update = 0.008
    n = 2  # Number of dimensions 
    #Node positions
    nodes = np.random.randn(num_nodes, n)
    #Node object array
    nodesObjects = []

    # Add measurment for each node yi = theta_t + v_i
    nodes_va = (50 * np.ones((num_nodes, 1))) + \
        (1 * np.random.randn(num_nodes, 1))
    nodes_va0 = nodes_va  # Save inital measurments

    #Populate node object array
    for i, item in enumerate(nodes, start=0):
        nodesObjects.append(GraphNode(i, np.array([nodes[i][0], nodes[i][1]]), []))

    #Populate node neighbors
    for i, item in enumerate(nodesObjects, start=0):
        nodesObjects[i].FindyourNeighbors(nodesObjects, r)

    #Print data
    for i, item in enumerate(nodesObjects, start=0):
        
        print("Node ",  nodesObjects[i].index, " X :", nodesObjects[i].position[0], " Y :", nodesObjects[i].position[1])
        print("- Has these neighbors ",  end = '')
        for j, item in enumerate(nodesObjects[i].neighbors, start=0):
            print(nodesObjects[i].neighbors[j]," ",  end = '')
        print("")
        print("")

    

    # Find the neighbors of the nodes
    #Neigbors = FindNeighbors(nodes, r, n, delta_t_update)

    DisplayGraph(nodesObjects)
    

    print("Data for target, robot and errhhor written")

# ----------------------------------------------------------------------------
# FUNCTION NAME:     DisplayGraph()
# PURPOSE:           Displays a graph give the nodes and neighbors
# -----------------------------------------------------------------------------
def DisplayGraph(nodesObjects):

    # Create graph object
    G = nx.Graph()

    # Display x and y coord
   # for i, item in enumerate(nodesObjects, start=0):
        #print("Node ", i, " X :", nodesObjects[i].position[0], " Y :", nodesObjects[i].position[1])

    # Add nodes
    for i, item in enumerate(nodesObjects, start=0):
        G.add_node(str(i), pos=(nodesObjects[i].position[0], nodesObjects[i].position[1]))

    # Add edges
    for i, item in enumerate(nodesObjects, start=0):
        for j, item in enumerate(nodesObjects[i].neighbors, start=0):
            G.add_edge(str(nodesObjects[i].index), str(nodesObjects[i].neighbors[j]))

    # Get subplots
    fig, ax = plt.subplots()

    # Draw graph object
    nx.draw(G, nx.get_node_attributes(G, 'pos'),
            with_labels=True, node_size=250)

    # Add x and y axis ticks and labels
    limits = plt.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Graph")
    plt.show()


# ----------------------------------------------------------------------------
# FUNCTION NAME:     FindNeighbors()
# PURPOSE:           Computes the neighbors given a nodes array
# -----------------------------------------------------------------------------
def FindNeighbors(nodes, r, n, delta_t):

    #Tempt array for neigbors
    Neigbor_array = []

    # Loop through each node and compare it to all other nodes
    for i, item in enumerate(nodes, start=0):
        for j, item in enumerate(nodes, start=0):

            # Calculate distance between node i and j
            neighborDistance = np.sqrt(
                np.square(nodes[j][0] - nodes[i][0]) + np.square(nodes[j][1] - nodes[i][1]))
            # If within range they are neighbors
            if(neighborDistance <= r and neighborDistance != 0):
                Neigbor_array.append(np.array([i, j]))
                print("Node ", i, "->", j)
                # print("Distance :" , np.sqrt(np.square(nodes[j][0] - nodes[i][0]) + np.square(nodes[j][1] - nodes[i][1])))

    return Neigbor_array


if __name__ == "__main__":
    main()
