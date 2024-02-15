
import algorithms_dev as ad
import benchmarks_dev as bd

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import array
import random
import sys
import numpy as np
import csv


cardinalities=[]

# Returns a random number in the range between 0..cardinality-1
def randomUnderCardinality(index):    
    return random.randint(0,cardinalities[index]-1)

def initRepeatWithCardinalities(container, func, n):
    return container(func(i) for i in range(n))

def main():
   
    with open('data-tree-eda.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        # Get the required parameters  from the first line of data-tree-eda.csv  
        row=next(readCSV)   
        nCheckboard=int(row[0])
        mCheckboard=int(row[1])
        sizesel=int(row[2])
        populSize=int(row[3])
        numgen=int(row[4])
        # Get the cardinalities from the second line of data-tree-eda.csv  
        row=next(readCSV)   
        for line in range(len(row)):
            cardinalities.append(int(row[line]))
    
    # Dimensions of the checkboard determine the size of the individual
    numberOfVariables = nCheckboard * mCheckboard

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Structure initializers
    toolbox.register("individual", initRepeatWithCardinalities, creator.Individual, randomUnderCardinality, numberOfVariables)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", bd.evalCheckboardNeighbours,nCB=nCheckboard,mCB=nCheckboard)
    toolbox.register("select", tools.selBest)

    random.seed(64)
    
    pop = toolbox.population(n=populSize)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = ad.treeEDA(pop,toolbox, sizesel,cardinalities,ngen=numgen,stats=stats,halloffame=hof, vbse=True)

    return pop, log, hof
 
if __name__ == "__main__":
    main()