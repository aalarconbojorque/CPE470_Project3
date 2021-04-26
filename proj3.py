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

    # #Print data
    # for i, item in enumerate(nodesObjects, start=0):
        
    #     # print("Node ",  nodesObjects[i].index, " X :", nodesObjects[i].position[0], " Y :", nodesObjects[i].position[1])
    #     # print("- Has these neighbors ",  end = '')
    #     for j, item in enumerate(nodesObjects[i].neighbors, start=0):
    #         #print(nodesObjects[i].neighbors[j]," ",  end = '')
    #     #print("")
    #     #print("")

    #Single Cell Location, calc QBAR
    Cell_Y_Sum = 0
    Cell_X_Sum = 0
    for i, item in enumerate(nodesObjects, start=0):
        Cell_Y_Sum = Cell_Y_Sum + nodesObjects[i].position[1]
        Cell_X_Sum = Cell_X_Sum + nodesObjects[i].position[0]  
    Cell_Y_Sum = (1/num_nodes) * Cell_Y_Sum
    Cell_X_Sum = (1/num_nodes) * Cell_X_Sum
    Q_Bar = np.array([Cell_X_Sum, Cell_Y_Sum])

   
    DisplayGraph(nodesObjects, Q_Bar)
    X_Values = []
    
    for t in range(1, 80):
        X_Values.insert(0, (0 * np.ones((num_nodes, 1))) + \
        (1 * np.ones((num_nodes, 1))))

    nodes_initial = nodes_va0
    X_Values.insert(0, nodes_initial)
    summat = 0
  
    #Computer in t range
    for t in range(1, 80):
        #For all nodes
        for i, item in enumerate(nodesObjects, start=0):
            summat = 0

            #Summation for all neighbors of i
            for j, item in enumerate(nodesObjects[i].neighbors, start=0):
                print(X_Values[t-1][nodesObjects[i].neighbors[j]])
                summat = summat + (WeightDesign1(i, nodesObjects[i].neighbors[j] , nodesObjects, num_nodes, Q_Bar) * X_Values[t-1][nodesObjects[i].neighbors[j]])

            wehgit = WeightDesign1(i,i, nodesObjects, num_nodes, Q_Bar)
            print(summat, float(X_Values[t-1][i]))
            val1_a =  wehgit * float(X_Values[t-1][i]) + summat

            print("Past X : ", X_Values[t-1][i] , " New X : ", val1_a)

            X_Values[t][i] =  val1_a

            new_NodePos =  Q_Bar - nodesObjects[i].position 
            nodesObjects[i].position = new_NodePos +  nodesObjects[i].position
            nodesObjects[i].FindyourNeighbors(nodesObjects, r)


            # neigDistance = np.sqrt(np.square(nodesObjects[i].position[0] - Q_Bar[0]) + np.square(nodesObjects[i].position[1] - Q_Bar[1]))
            # print(neigDistance)

            # if((neigDistance <= 0.5 and neigDistance >= 0.5)):
            #     nodesObjects[i].position = new_NodePos
            #     nodesObjects[i].FindyourNeighbors(nodesObjects, r)
        


    # for i, item in enumerate(nodes_va0, start=0):
    #     print(nodes_va0[i])

    # print("EF")
    # for i, item in enumerate(X_Values[79], start=0):
    #     print(X_Values[79][i])

    DisplayGraph(nodesObjects, Q_Bar) 

    x_a = []
    y_a = []

    for i, item in enumerate(nodesObjects, start=0):
        x_a.append(i)
        y_a.append(X_Values[0][i])

    plt.scatter(x_a, y_a, s=20, alpha=1)

    x_a = []
    y_a = []

    for i, item in enumerate(nodesObjects, start=0):
        x_a.append(i)
        y_a.append(X_Values[79][i])

    
    plt.scatter(x_a, y_a, s=20, alpha=1)
    plt.show()


    print("Data ")


def WeightDesign1(i, j, nodesObjects, num_nodes, Q_Bar):
    
    cv = 0.001
    ris = 1.6
    
    c1W = ((1/2)*(2*cv)/((ris**2)*(num_nodes-1)))
    Equalsum = 0

    #Weighted average design 1 if i != j
    if(i != j):

        Vi = V_t(nodesObjects, i, Q_Bar)
        Vj = V_t(nodesObjects, j, Q_Bar)

        if(Vi == 0 or Vj == 0):
            return 0
        
        else:
            return (c1W/(Vi +Vj))
    
    #Weighted average design 1 if i == j    
    else:
        #Calculate weights for each neighobr of node i, sum them
        for k, item in enumerate(nodesObjects[i].neighbors, start=0):
            if(i != nodesObjects[i].neighbors[k]):
                Equalsum = Equalsum + WeightDesign1(i, nodesObjects[i].neighbors[k], nodesObjects, num_nodes, Q_Bar)
            else:
                Equalsum = Equalsum

        return 1 - Equalsum


# ----------------------------------------------------------------------------
# FUNCTION NAME:     V_t()
# PURPOSE:           Calculate noise variance given a node i
# -----------------------------------------------------------------------------
def V_t(nodesObjects, i, Q_Bar):

    #Calculate noise variance
    cv = 0.001
    ris = 1.6

    Node_Pos = nodesObjects[i].position
    Distance = Node_Pos - Q_Bar
    Distance = np.linalg.norm(Distance)

    if(Distance <= ris):
        Distance = Distance**2
        V_i = Distance + cv
        V_i = (V_i/(ris**2))
        return V_i
    else :
        return 0




# ----------------------------------------------------------------------------
# FUNCTION NAME:     DisplayGraph()
# PURPOSE:           Displays a graph give the nodes and neighbors
# -----------------------------------------------------------------------------
def DisplayGraph(nodesObjects, Q_Bar):

    # Create graph object
    G = nx.Graph()

    # Display x and y coord
   # for i, item in enumerate(nodesObjects, start=0):
        #print("Node ", i, " X :", nodesObjects[i].position[0], " Y :", nodesObjects[i].position[1])

    # Add nodes
    for i, item in enumerate(nodesObjects, start=0):
        G.add_node(str(i), pos=(nodesObjects[i].position[0], nodesObjects[i].position[1]))
    
    G.add_node("C", pos=(Q_Bar[0], Q_Bar[1]))

    # Add edges
    for i, item in enumerate(nodesObjects, start=0):
        for j, item in enumerate(nodesObjects[i].neighbors, start=0):
            G.add_edge(str(nodesObjects[i].index), str(nodesObjects[i].neighbors[j]))

    # Get subplots
    fig, ax = plt.subplots()

    #Color cell
    color_map = []
    for node in G:
        if node == "C":
            color_map.append('red')
        else: 
            color_map.append('#00b4d9') 

    # Draw graph object
    nx.draw(G, nx.get_node_attributes(G, 'pos'),
            with_labels=True, node_size=250, node_color=color_map)

    # Add x and y axis ticks and labels
    limits = plt.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Graph")
    plt.show()


# # ----------------------------------------------------------------------------
# # FUNCTION NAME:     FindNeighbors()
# # PURPOSE:           Computes the neighbors given a nodes array
# # -----------------------------------------------------------------------------
# def FindNeighbors(nodes, r, n, delta_t):

#     #Tempt array for neigbors
#     Neigbor_array = []

#     # Loop through each node and compare it to all other nodes
#     for i, item in enumerate(nodes, start=0):
#         for j, item in enumerate(nodes, start=0):

#             # Calculate distance between node i and j
#             neighborDistance = np.sqrt(
#                 np.square(nodes[j][0] - nodes[i][0]) + np.square(nodes[j][1] - nodes[i][1]))
#             # If within range they are neighbors
#             if(neighborDistance <= r and neighborDistance != 0):
#                 Neigbor_array.append(np.array([i, j]))
#                 print("Node ", i, "->", j)
#                 # print("Distance :" , np.sqrt(np.square(nodes[j][0] - nodes[i][0]) + np.square(nodes[j][1] - nodes[i][1])))

#     return Neigbor_array


if __name__ == "__main__":
    main()
