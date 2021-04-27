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
# Andy Alarcon       04-26-2021     1.1 ... implemented weight design 1, graphs for average and measurements
# Andy Alarcon       04-27-2021     1.1 ... adjusted graphs, added weight design 2
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random
import copy

#Class that represent a single graph node
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
    #Iterations
    it = 500

    # Add measurment for each node yi = theta_t + v_i
    nodes_va = (50 * np.ones((num_nodes, 1))) + \
        (1 * np.random.randn(num_nodes, 1))

    #Populate node object array
    for i, item in enumerate(nodes, start=0):
        nodesObjects.append(GraphNode(i, np.array([nodes[i][0], nodes[i][1]]), []))

    #Populate node neighbors
    for i, item in enumerate(nodesObjects, start=0):
        nodesObjects[i].FindyourNeighbors(nodesObjects, r)

    #Copy node Objects
    InitalNodeObjects = copy.deepcopy(nodesObjects)

    #Single Cell Location, calc QBAR
    Cell_Y_Sum = 0
    Cell_X_Sum = 0
    for i, item in enumerate(nodesObjects, start=0):
        Cell_Y_Sum = Cell_Y_Sum + nodesObjects[i].position[1]
        Cell_X_Sum = Cell_X_Sum + nodesObjects[i].position[0]
    Cell_Y_Sum = (1/num_nodes) * Cell_Y_Sum
    Cell_X_Sum = (1/num_nodes) * Cell_X_Sum
    Q_Bar = np.array([Cell_X_Sum, Cell_Y_Sum])

    #Assign initial measurement weight 1 and 2
    cv = 0.01
    rs = 1.6
    v = 0
    n = 0
    F = 50
    for i, item in enumerate(nodes_va, start=0):
        v = ((np.linalg.norm(nodesObjects[i].position-Q_Bar)**2)+cv)/(rs**2)
        n = np.random.normal(0.0,v)
        nodes_va[i] = F + n
        v = 0
        n = 0

    # Save inital measurments
    nodes_va0 = nodes_va  

    #Display inital graph
    DisplayGraph(nodesObjects, Q_Bar, "Graph.png")


    # Weight Design 1
    #--------------------------------------------------------------------------------------------------------------------------

    #Initalize x(t) array for measurement
    X_Values = []
    E_Values = []
    for t in range(1, it):
        X_Values.insert(0, (0 * np.ones((num_nodes, 1))) + \
        (1 * np.ones((num_nodes, 1))))

        E_Values.insert(0, (0 * np.ones((num_nodes, 1))) + \
        (1 * np.ones((num_nodes, 1))))

    #Insert initial measurement
    nodes_initial = nodes_va0
    X_Values.insert(0, nodes_initial)
    E_Values.insert(0, nodes_initial)
    summat = 0

    
    for t in range(1, it):

        #For all nodes
        for i, item in enumerate(nodesObjects, start=0):

            #Reset summation of neighbor weights
            summat = 0

            #Summation for all neighbors of node i
            for j, item in enumerate(nodesObjects[i].neighbors, start=0):
                summat = summat + (WeightDesign1(i, nodesObjects[i].neighbors[j] , nodesObjects, num_nodes, Q_Bar) * X_Values[t-1][nodesObjects[i].neighbors[j]])

            #Compute weight for ii
            iiWeightComp = WeightDesign1(i,i, nodesObjects, num_nodes, Q_Bar)
            val1 =  iiWeightComp * float(X_Values[t-1][i]) + summat
            
            #Assign next measurment
            if(math.isnan(val1)):
                out_num = np.nan_to_num(val1)
                X_Values[t][i] =  out_num
            else:
                X_Values[t][i] = val1

            #Compute weighted average, correct if becomes nan        
            np.seterr(divide='ignore', invalid='ignore')
            PossibleNan = ((iiWeightComp * X_Values[t-1][i]) /(iiWeightComp))
            if(math.isnan(PossibleNan)):
                out_num = np.nan_to_num(PossibleNan)
                E_Values[t][i] = out_num
            
            else:
                E_Values[t][i] = PossibleNan

            #Move current node towards cell
            new_NodePos =  Q_Bar - nodesObjects[i].position
            nodesObjects[i].position = new_NodePos +  nodesObjects[i].position
            nodesObjects[i].FindyourNeighbors(nodesObjects, r)
        
    DisplayScatterPlot(nodesObjects, X_Values, it, 'WeightedDesign1_NodesScatterPlot.png')
    DisplayNodesGraph(E_Values, X_Values, it, 'WeightedDesign1_NodesDistance.png')

    # Weight Design 2
    #--------------------------------------------------------------------------------------------------------------------------
  
    #Display inital graph
    it = 40
    DisplayGraph(InitalNodeObjects, Q_Bar, "Graph2.png")

    #Reassign inital node object
    nodesObjects = copy.deepcopy(InitalNodeObjects)

    #Initalize x(t) and E(t) array for measurement
    X2_Values = []
    E2_Values = []
    for t in range(1, it):
        X2_Values.insert(0, (0 * np.ones((num_nodes, 1))) + \
        (1 * np.ones((num_nodes, 1))))

        E2_Values.insert(0, (0 * np.ones((num_nodes, 1))) + \
        (1 * np.ones((num_nodes, 1))))

    #Insert initial measurement
    nodes_initial = nodes_va0
    X2_Values.insert(0, nodes_initial)
    E2_Values.insert(0, nodes_initial)
    summat = 0

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

            #Compute weighted average, correct if becomes nan        
            np.seterr(divide='ignore', invalid='ignore')
            PossibleNan = ((iiWeightComp * X2_Values[t-1][i]) /(iiWeightComp))
            if(math.isnan(PossibleNan)):
                out_num = np.nan_to_num(PossibleNan)
                E2_Values[t][i] = out_num
            
            else:
                E2_Values[t][i] = PossibleNan

            #Move current node towards cell
            new_NodePos =  Q_Bar - nodesObjects[i].position
            nodesObjects[i].position = new_NodePos +  nodesObjects[i].position
            nodesObjects[i].FindyourNeighbors(nodesObjects, r)
        
    DisplayScatterPlot(nodesObjects, X2_Values, it, 'WeightedDesign2_NodesScatterPlot.png')
    DisplayNodesGraph(E2_Values, X2_Values, it, 'WeightedDesign2_NodesDistance.png')

    print("Graph images for weighted design 1 and 2 created")

# ----------------------------------------------------------------------------
# FUNCTION NAME:     WeightDesign2()
# PURPOSE:           Calculate weight design 2 given a node i and j
# -----------------------------------------------------------------------------
def WeightDesign2(i, j, nodesObjects, num_nodes, Q_Bar):

    cv = 0.001
    ris = 1.6
    c2W = (0.01)*(cv/ris**2)

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
# FUNCTION NAME:     WeightDesign1()
# PURPOSE:           Calculate weight design 1 given a node i and j
# -----------------------------------------------------------------------------
def WeightDesign1(i, j, nodesObjects, num_nodes, Q_Bar):

    cv = 0.001
    ris = 1.6
    Equalsum = 0

    #Weighted average design 1 if i != j
    if(i != j and j in nodesObjects[i].neighbors):
        
        ni = len(nodesObjects[i].neighbors)
        ni = ni - 1
        c1W = ((0.01)*(2*cv)/((ris**2)*(np.absolute(ni))))
        Vi = V_t(nodesObjects, i, Q_Bar)
        Vj = V_t(nodesObjects, j, Q_Bar)

        if(Vi == 0 or Vj == 0):
            return 0

        else:
            return (c1W/(Vi +Vj))

    #Weighted average design 1 if i == j
    elif (i == j):
        #Calculate weights for each neighobr of node i, sum them
        for k, item in enumerate(nodesObjects[i].neighbors, start=0):
                Equalsum = Equalsum + WeightDesign1(i, nodesObjects[i].neighbors[k], nodesObjects, num_nodes, Q_Bar)

        return 1 - Equalsum
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
