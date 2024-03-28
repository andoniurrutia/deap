
#from nose import with_setup
#import unittest
import algorithms_dev as ad
from sklearn.metrics import mutual_info_score
import numpy as np



def test_init():
# This test is for visualizing the structure of the frequency matrix and how it is initialized by the first parameter(in this case 1000)
# of the init() function. As we don't want to store twice the value of every pair, the strucure of the frequency matrix for a 
# 4-variable case will be: (0,1)->[x x x x],(0,2)->[x x x x],(0,3)->[x x x x],(1,2)->[x x x x],(1,3)->[x x x x] and (2,3)->[x x x x].   
# In this example, as we dont execute addFrequenciestoMatrix(), the data of the initial population is ignored.
    cardinalities=[2,2,2,3]
    population=[[1,0,1,0],[1,0,1,0],[1,0,1,0],[0,1,1,1],[0,1,0,1],[1,1,1,1],[0,0,1,0],[1,1,1,1]]
    ad.init(0.0,len(population[0]),population,cardinalities)
    print("INITIAL FREQUENCY MATRIX:\n"+str(ad.df) + "\n")

def test_mutualInformation():
# This test verifies that the getMutualInfo() function returns correct values, comparing them with the ones
# by the sklearn library

    cardinalities=[2,2,2,3]
    population=[[1,0,1,0],[1,0,1,0],[1,0,1,0],[0,1,1,1],[0,1,0,1],[1,1,1,1],[0,0,1,0],[1,1,1,1]]
    ad.init(0.0,len(population[0]),population,cardinalities)
    ad.addFrequenciestoMatrix(population)
    print("Mutual information with our algorithm:"+str(ad.getMutualInfo(1,2)))
    print("Mutual information with SKLearn method:"+str(mutual_info_score(np.transpose(population)[1],np.transpose(population)[2])))
    print("\n")

def test_conditionalFrequencies():
# This test verifies that the getConditionalFrequency()) function returns correct values. The first parameter of the function
# is the variable for which we want to get the frequency. The second parameter marks the variable for which we know the value. 
# The third parameter is the known value:0 or 1.

    cardinalities=[2,2,2,3]
    population=[[0,0,1,0],[1,0,1,0],[0,1,1,0],[0,1,0,1],[1,0,1,1],[1,1,1,0],[1,1,1,1],[1,1,1,1]]
    ad.init(0.0,len(population[0]),population,cardinalities)
    ad.addFrequenciestoMatrix(population)

    # When x1=1 the normalized population=[[0,1,x,x],[0,1,x,x],[1,1,x,x],[1,1,x,x],[1,1,x,x]]-> x0 is 0 in 2 out of 5  So it should return  [0.4,0.6]
    print ("Conditional frequencies for X0=0 and X0=1 values  when X1=1:"+str(ad.getConditionalFrequency(0,1,0,1))+"=[0.4,0.6]")
   
    #When x1=0 the normalized population=[[0,0,1,0][1,0,1,0],[1,0,1,1]]-> x0 is 0 in 1 out of 3  So it should return  [0.333333,0.666666]
    print ("Conditional frequencies for X0=0 and X0=1 values  when X1=0:"+str(ad.getConditionalFrequency(0,1,0,0))+"=[0.333333,0.666666]")
    
    #When x1=0 the normalized population=[[x,0,1,0],[x,0,1,0],[x,0,1,1]]-> all the x2 values are 0. So it should return  [0.0,1.0]
    print ("Conditional frequencies for X2=0 and X2=1 values  when X1=0:"+str(ad.getConditionalFrequency(2,1,0,0))+"=[0.0,1.0]")
    
    #When x1=1 the normalized population=[[0,1,1,0],[0,1,0,1],[1,1,1,0],[1,1,1,1],[1,1,1,1]]-> x2 is 0 in 1 out of 5 . So it should return  [0.2,0.8]
    print ("Conditional frequencies for X2=0 and X2=1 values  when X1=1:"+str(ad.getConditionalFrequency(2,1,0,1))+"=[0.2,0.8]")
   

def test_createMSPfromMI():
# This test prompts the structure of the whole Mutual Information Graph. Then it prompts a subset of the latter, which forms the Minimum Spanning 
# tree(with negative vaules for maximizing the MI). Then it prompts the bits which are generated while traversing the tree
    
    cardinalities=[2,2,2,3]
    population=[[0,0,1,0],[0,0,1,0],[0,1,1,0],[0,1,0,1],[1,0,1,1],[1,0,1,0],[1,1,1,1],[1,1,1,1]]
    #population=[[0,0,1,0,0,0,1,0],[0,1,1,0,0,1,0,1],[1,0,1,1,1,0,1,0],[1,1,1,1,1,1,1,1]]
    ad.init(0.0,len(population[0]),population,cardinalities)

    ad.addFrequenciestoMatrix(population)
    T=ad.createMSPfromMI(verbose=True)
    for i in range(len(population)):
        ad.traverseTree(T,vbse=True)


    
test_init()
test_mutualInformation()
test_conditionalFrequencies()
test_createMSPfromMI()

    