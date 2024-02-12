#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

n=int(input("n:"))
m=int(input("m:"))
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, int(n)*int(m))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalCheckboardNeighbours(individual):
    #Example myArray=[[0,1,0,0],[0,1,1,0],[0,1,0,1],[0,1,0,0]]
    myArray=np.reshape(individual, (-1, m))

    counter=0
    for i,row in enumerate(myArray):
        for j,entry in enumerate(row):
            if (i>0 and entry!=myArray[i-1][j]):
                counter=counter+1
            if (i<n-1 and entry!=myArray[i+1][j]):
                counter=counter+1
            if (j>0 and entry!=myArray[i][j-1]):
                counter=counter+1
            if (j<m-1 and entry!=myArray[i][j+1]):
                counter=counter+1
    return counter,
    

toolbox.register("evaluate", evalCheckboardNeighbours)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof

if __name__ == "__main__":
    main()