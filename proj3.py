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
import numpy as np
import math
import random


def main():
    r = 1 #Set Communication Range
    num_nodes = 10 #Randomly generated nodes
    delta_t_update = 0.008 
    n = 2 #Number of dimensions
    nodes = np.random.randn(num_nodes, n)

    #Add measurment for each node yi = theta_t + v_i
    nodes_va = (50 * np.ones((num_nodes, 1))) + (1 * np.random.randn(num_nodes, 1))
    nodes_va0 = nodes_va #Save inital measurments

    #Find the neighbors of the nodes
    Neigbors = FindNeighbors(nodes, r , n, delta_t_update)

    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()
    print("Data for target, robot and errhhor written")
    

    

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
                print("Node " , i , "->" , j)
                print("Distance :" , np.sqrt(np.square(nodes[j][0] - nodes[i][0]) + np.square(nodes[j][1] - nodes[i][1])))

    return Neigbor_array

    
if __name__ == "__main__":
    main()