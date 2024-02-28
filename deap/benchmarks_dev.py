
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
        Value of the Rastrigin function at point x.
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    """
    Calculate the value of Rosenbrock function for a given point x.
    
    Parameters:
    x : numpy array
        Input vector.
        
    Returns:
    float
        Rosenbrock function at point x.
    """
    value = 0
    for i in range(len(x) - 1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value

def griewank_function(x):
    """
    Griewank Function

    Parameters:
    x : numpy array
        Input vector.

    Returns:
    float
        Value of the function at point x.
    """
    n = len(x)
    sum_value = np.sum(x**2) / 4000
    prod_value = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    return sum_value - prod_value + 1

def michalewicz_function(x, m=10):
    """
    Michalewicz function.

    Parameters:
    x : numpy array
        Input vector.
    m : int, optional
        The parameter "m" controls the steepness of the valleys.

    Returns:
    float
        Value of the function at point x.
    """
    n = len(x)
    return -np.sum(np.sin(x) * (np.sin(((np.arange(1, n + 1) * x**2) / np.pi))**(2 * m)))