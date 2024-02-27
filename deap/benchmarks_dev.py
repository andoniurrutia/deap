
import numpy as np

def evalCheckboardNeighbours(individual,nCB,mCB):
    
    checkboard=np.reshape(individual, (-1, mCB))
    counter=0
    for i,row in enumerate(checkboard):
        for j,entry in enumerate(row):
            if (i>0 and entry!=checkboard[i-1][j]):
                counter=counter+1
            if (i<nCB-1 and entry!=checkboard[i+1][j]):
                counter=counter+1
            if (j>0 and entry!=checkboard[i][j-1]):
                counter=counter+1
            if (j<mCB-1 and entry!=checkboard[i][j+1]):
                counter=counter+1
    return counter,

def rastrigin(x):
    """
    Rastrigin Function

    Parameters:
    x : numpy array
        Input vector.

    Returns:
    float
        VAlue of the Rastrigin function at point x.
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))