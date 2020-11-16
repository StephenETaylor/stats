#!/usr/bin/env python3
"""
    read in a file of variables and correlate each column with the others
    variables separated by whitespace, so tab-separated .CSV files work well

"""
import math
import numpy as np
import scipy.stats as ss
import sys

def main():

    # read in array
    ma0 = []
    for lin in sys.stdin:
        line = lin.strip().split()
        lfl = [float(x) for x in line]
        ma0.append(lfl)

    #convert to numpy ndarray
    mat = np.array(ma0)

    #do a bunch of spearman correlations
    rho, p = ss.spearmanr(mat)

    print(rho)
    print(p)



    


main()
