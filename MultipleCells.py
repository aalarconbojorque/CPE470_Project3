# -----------------------------------------------------------------------------
# FILE NAME:         WeightedAverageConsensus.py
# USAGE:             python3 WeightedAverageConsensus.py
# NOTES:             Requires matplotlib and numpy installation
#                    Requires networkx installation
#                    Requires Python3
#
# MODIFICATION HISTORY:
# Author             Date           Modification(s)
# ----------------   -----------    ---------------
# Andy Alarcon       04-23-2021     1.0 ... Setup dev environment, imported NumPy
# Andy Alarcon       04-25-2021     1.1 ... imported matpotlib and networkx, created graph display
# Andy Alarcon       04-26-2021     1.2 ... implemented weight design 1, graphs for average and measurements
# Andy Alarcon       04-27-2021     1.3 ... adjusted graphs, added weight design 2
# Andy Alarcon       04-28-2021     1.3 ... Fixed bug that provided inaccurate neighbors
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random
import copy
import fileinput 

#Class that represent a single graph node
class GraphNode:
    def __init__(self, index, position, neighbors):
        self.index = index
        self.position = position
        self.neighbors = neighbors

    def FindyourNeighbors(self, nodesObjects, r):
      self.neighbors = []  
      for j, item in enumerate(nodesObjects, start=0):
            # Calculate distance between self and j
            neighborDistance = np.sqrt(np.square(nodesObjects[j].position[0] - self.position[0]) + np.square(nodesObjects[j].position[1] - self.position[1]))
            # If within range they are neighbors
            if(neighborDistance <= r and neighborDistance != 0):
                self.neighbors.append(j)
               # print("Node ", self.index, "->", j)

#Class that represent a cell node
class Cell:
    def __init__(self, initalMeasurement):    
        self.initalMeasurement = initalMeasurement

    def FindyourNeighbors(self, nodesObjects, r):
      pass


def main():
    r = 8  # Set Communication Range
    num_nodes = 10  # Randomly generated nodes
    delta_t_update = 0.008
    n = 2  # Number of dimensions
    it = 80
    
    #Input Scalar Field values 25X25 matrix
    r, c = (25, 25) 
    Cells = [[0 for i in range(c)] for j in range(r)] 
    Results_Cells = [[0 for i in range(c)] for j in range(r)]  
    rowVal = 24
    ColVal = 0
    file = open('Scalar_Field_data.txt','r', encoding='utf-16-le')
    lines = file.readlines()
    for line in lines:
        row = line.strip().split('|')
        for col in row:
            col = col.replace("\ufeff", "")
            Cells[rowVal][ColVal] = float(col)
            ColVal = ColVal + 1
        ColVal = 0
        rowVal = rowVal - 1


    #Set node positions between low and high-1
    nodes = np.random.randint(low=0 , high=26, size=(num_nodes, n))
    #Node object array
    nodesObjects = []
   
    # Add measurment for each node yi = theta_t + v_i
    nodes_va = (50 * np.ones((num_nodes, 1))) + \
        (1 * np.random.randn(num_nodes, 1))
    
    #Populate node object array
    for i, item in enumerate(nodes, start=0):
        nodesObjects.append(GraphNode(i, np.array([nodes[i][0], nodes[i][1]]), []))

    #Populate node neighbors
    for i, item in enumerate(nodesObjects, start=0):
        nodesObjects[i].FindyourNeighbors(nodesObjects, r)

    

    DisplayGraph(nodesObjects, np.array([5,5]), "Graph.png")

    #Traverse the 25x25 matrix
    for row, item in enumerate(Cells, start=0):

        for col, item in enumerate(Cells[row], start=0):
            
            #Initial Cell Value
            Initial_CellVal = Cells[row][col]
            Q_Bar = np.array([row,col])

            # Add measurment for each node yi = theta_t + v_i
            nodes_va = (Initial_CellVal* np.ones((num_nodes, 1))) + \
            (1 * np.random.randn(num_nodes, 1))

            #Initalize x(t) array for measurement
            X_Values = []

            for t in range(1, it):
                X_Values.insert(0, (0 * np.ones((num_nodes, 1))) + \
                (1 * np.ones((num_nodes, 1))))  

            X_Values.insert(0, nodes_va)
            summat = 0

            #Begin consensus
            for t in range(1, it):
                #For all nodes
                for i, item in enumerate(nodesObjects, start=0):
                    
                    #Reset summation of neighbor weights
                    summat = 0

                    #Summation for all neighbors of node i
                    for j, item in enumerate(nodesObjects[i].neighbors, start=0):
                        summat = summat + (WeightDesign2(i, nodesObjects[i].neighbors[j] , nodesObjects, num_nodes, Q_Bar) * X2_Values[t-1][nodesObjects[i].neighbors[j]])
                    
                    #Compute weight for ii
                    iiWeightComp = WeightDesign2(i,i, nodesObjects, num_nodes, Q_Bar)
                    val1 =  iiWeightComp * float(X2_Values[t-1][i]) + summat

                    #Assign next measurment
                    X2_Values[t][i] =  val1

                    #Move current node towards cell
                    #Compute relative y pos 
                    yrv = Q_Bar[1] - nodesObjects[i].position[1]
                    #Compute relative x pos 
                    xrv = Q_Bar[0] - nodesObjects[i].position[0]
                    #compute direction
                    Hrv = np.arctan2(yrv, xrv)
                    #Compute distance
                    Diastance = np.sqrt(np.square(Q_Bar[0] - nodesObjects[i].position[0]) + np.square(Q_Bar[1] - nodesObjects[i].position[1]))
                    if 0.1 <= Diastance <= 0.2:
                        nodesObjects[i].FindyourNeighbors(nodesObjects, r)

                    else:
                        new_NodePos = nodesObjects[i].position + (0.005 * np.array([np.cos(Hrv), np.sin(Hrv)]))
                        nodesObjects[i].position = new_NodePos
                        nodesObjects[i].FindyourNeighbors(nodesObjects, 2)
                    







    #GraphField(r, c, Cells, "Initial_ScalarField", "1st Field")

    print("Graph images for weighted design 1 and 2 created")

# ----------------------------------------------------------------------------
# FUNCTION NAME:     DisplayNodesGraph()
# PURPOSE:           Displays Plot for nodes
# -----------------------------------------------------------------------------
def GraphField(r, c, Cells, FileName, Title):

    row_labels = range(r)
    col_labels = range(c)
    plt.matshow(Cells, extent=[0, 25, 25, 0])
    plt.xticks(range(c), col_labels)
    plt.yticks(range(r), row_labels)
    plt.colorbar()
    figure = plt.gcf()
    figure.set_size_inches(9, 9)
    plt.grid(color="black")
    plt.title(Title)
    plt.savefig(FileName, dpi=1200) 
    plt.show()



# ----------------------------------------------------------------------------
# FUNCTION NAME:     WeightDesign2()
# PURPOSE:           Calculate weight design 2 given a node i and j
# -----------------------------------------------------------------------------
def WeightDesign2(i, j, nodesObjects, num_nodes, Q_Bar):

    cv = 0.001
    ris = 2
    c2W = (0.5)*((cv)/(ris**2))

    Equalsum = 0
    #WeightDesign2 if i != j
    if(i != j and j in nodesObjects[i].neighbors):
        ans = 1 - WeightDesign2(i, i, nodesObjects, num_nodes, Q_Bar)
        ans = (ans/np.absolute(len(nodesObjects[i].neighbors)))
        return ans
    #WeightDesign2 design 2 if i == j
    elif (i == j):
        Vi = V_t(nodesObjects, i, Q_Bar)
        if(Vi == 0):
            return 0
        ans = (c2W/Vi)
        return ans
    else:
        return 0


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
# FUNCTION NAME:     DisplayNodesGraph()
# PURPOSE:           Displays Plot for nodes
# -----------------------------------------------------------------------------
def DisplayNodesGraph(E_Values, X_Values, it, FileName):

    for t, item in enumerate(E_Values, start=0):
        E_Values[t] = X_Values[t] - E_Values[t]

    x_a = []
    y_a = []

    for i, item in enumerate(E_Values[0], start=0):
        y_a = []
        x_a = []
        for t in range(3, it - 1):
            x_a.append(t)
            y_a.append(E_Values[t][i])
        plt.plot(x_a, y_a, label=str(i))

    #plt.title("Average Comparison")  
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    plt.savefig(FileName, dpi=1200) 
    plt.show()
    plt.close()

# ----------------------------------------------------------------------------
# FUNCTION NAME:     DisplayScatterPlot()
# PURPOSE:           Displays Scatter Plot for measurements
# -----------------------------------------------------------------------------
def DisplayScatterPlot(nodesObjects, X_Values, it, FileName):

    x_a = []
    y_a = []

    for i, item in enumerate(nodesObjects, start=0):
        x_a.append(i)
        y_a.append(X_Values[0][i])

    plt.scatter(x_a, y_a, label="Inital Measurement")
    plt.plot(x_a, y_a)

    x_a = []
    y_a = []

    for i, item in enumerate(nodesObjects, start=0):
        x_a.append(i)
        y_a.append(X_Values[it - 1][i])

    #plt.title("Measurements Comparison")
    plt.scatter(x_a, y_a, label="Final Measurement")
    plt.plot(x_a, y_a)
    plt.xlabel("10 nodes")
    plt.ylabel("Value")
    plt.xticks(x_a)
    plt.legend(loc="best")

    plt.savefig(FileName, dpi=1200)  
    plt.show()


# ----------------------------------------------------------------------------
# FUNCTION NAME:     DisplayGraph()
# PURPOSE:           Displays a graph give the nodes and neighbors
# -----------------------------------------------------------------------------
def DisplayGraph(nodesObjects, Q_Bar, FileName):

    # Create graph object
    G = nx.Graph()


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
            color_map.append('#ADD8E6')

    # Draw graph object
    nx.draw(G, nx.get_node_attributes(G, 'pos'),
            with_labels=True, node_size=200, node_color=color_map)

    # Add x and y axis ticks and labels
    limits = plt.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel('X (pos)')
    plt.ylabel('Y (pos)')
    plt.title("10 Graph nodes")
    plt.savefig(FileName, dpi=1200) 
    plt.show()




if __name__ == "__main__":
    main()
