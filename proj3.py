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

    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()
    print("Data for target, robot and errhhor written")
    

    

# ----------------------------------------------------------------------------
# FUNCTION NAME:     CompMagSqr()
# PURPOSE:           Computes the magnitude squared
# -----------------------------------------------------------------------------

    
if __name__ == "__main__":
    main()