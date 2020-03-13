'''
Created on Nov 13, 2019

@author: Rocil
'''

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from numpy import empty

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def reLU(x):
    return max(0,x)

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def q1(x1, x2, w1, w2, wo, b, b_output):
    #     o1 = b[0] + (w1[0] * x1) + (w2[0] * x2)
    #     o2 = b[1] + (w1[1] * x1) + (w2[1] * x2)
    #     o3 = b[2] + (w1[2] * x1) + (w2[2] * x2)
    #     o4 = b[3] + (w1[3] * x1) + (w2[3] * x2)
    #     o5 = b[4] + (w1[4] * x1) + (w2[4] * x2)
    #     o6 = b[5] + (w1[5] * x1) + (w2[5] * x2)
    #     o7 = b[6] + (w1[6] * x1) + (w2[6] * x2)
    #     o8 = b[7] + (w1[7] * x1) + (w2[7] * x2)
    #     o9 = b[8] + (w1[8] * x1) + (w2[8] * x2)
    #     o10 = b[9] + (w1[9] * x1) + (w2[9] * x2)
    o = []
    for i in range(10):
        o.append(reLU(b[i] + w1[i] * x1 + w2[i] * x2))
    #print("[o1,o2,o3,o4,o5,o6,o7,o8,o9,o10] = ",o)
    y = b_output
    for i in range(len(o)):
        y = y + (wo[i] * o[i])
    #print("Input:", x1,",", x2, "    Output:", sigmoid(y))
    return sigmoid(y)

def q3(x1, x2, w11, w12, w21, w22, wo, b1, b2, bo):
    #     o1 = b[0] + (w1[0] * x1) + (w2[0] * x2)
    #     o2 = b[1] + (w1[1] * x1) + (w2[1] * x2)
    #     o3 = b[2] + (w1[2] * x1) + (w2[2] * x2)
    #     o4 = b[3] + (w1[3] * x1) + (w2[3] * x2)
    #     o5 = b[4] + (w1[4] * x1) + (w2[4] * x2)
    #     o6 = b[5] + (w1[5] * x1) + (w2[5] * x2)
    #     o7 = b[6] + (w1[6] * x1) + (w2[6] * x2)
    #     o8 = b[7] + (w1[7] * x1) + (w2[7] * x2)
    #     o9 = b[8] + (w1[8] * x1) + (w2[8] * x2)
    #     o10 = b[9] + (w1[9] * x1) + (w2[9] * x2)
    layer_output = empty([5,2])
    layer_input = empty([5,2])
    layer_input[0,:] = [x1,x2]
    for i in range(5):
        layer_output[i, :] = [reLU(b1[i] + layer_input[i,0] * w11[i] + layer_input[i,1]* w21[i] ), reLU(b2[i] + layer_input[i,0] * w21[i] + layer_input[i,1]* w22[i])]
        if((i+1) < 5 ):
            next_index = i+1
            layer_input[next_index,:] = layer_output[i,:]
        #print("Hidden Layer", i+1, "output: ", layer_output[i, :])

    y = bo + layer_output[4,0] * wo1[0] + layer_output[4,1] * wo1[1]
    #print("Input:", x1,",", x2, "    Final Output:", sigmoid(y))
    return sigmoid(y)
        
def feedforward(self,x,y,b):
    for i in range (0, 10):
        self.input[i] = [b[i],x[0],x[1]]
    for i in range(0,10):
        self.inputweights[i] = [1, self.weights[i], self.weights[i+1]]
            
    for i in range(0,10):
        self.h1[i] = reLU(np.dot(self.input[i],self.weights[i]))
            
    #self.weights0 = [1,self.weights[0],self.weights[1]]
    #self.weights1 = [1,self.weights[2],self.weights[3]] 
    #self.weights2 = [1,self.weights[4],self.weights[5]]
    #self.weights3 = [1,self.weights[6],self.weights[7]]
    #self.weights4 = [1,self.weights[8],self.weights[9]]
    #self.weights5 = [1,self.weights[10],self.weights[11]]
    #self.weights6 = [1,self.weights[12],self.weights[13]]
    #self.weights7 = [1,self.weights[14],self.weights[15]]
    #self.weights8 = [1,self.weights[16],self.weights[17]]
    #self.weights9 = [1,self.weights[18],self.weights[19]]
    self.x = x
    self.y = y
    self.inputA = [b[0],self.x[0],self.x[1]]
    self.inputB = [1,self.x[0],self.x[1]]
    self.output     = np.zeros(self.y.shape)
    self.u_A = np.dot(self.inputA, self.weightsA)
    self.layerA = reLU(self.u_A)
    self.v_A = self.layerA
    self.u_B = np.dot(self.inputB, self.weightsB)
    self.layerB = reLU(self.u_B)
    self.v_B = self.layerB
    self.inputC = [1, self.layerA, self.layerB]
    self.u_C = np.dot(self.inputC, self.weightsC)
    self.output = sigmoid(self.u_C)
    self.v_C = self.output
        #print("Question1: Weights: ", self.weights1, "Input:", x, self.u_A, " ", self.v_A," ", self.u_B, " ", self.v_B, " ", self.u_C, " ", self.v_C)
        
if __name__ == "__main__":
    
    #x1 = -1
    #x2 = -1
    wo = np.zeros(10)
    w1 = np.zeros(10)
    w2 = np.zeros(10)
    b = np.zeros(10)
    #wo = [1] * 10
    #w1 = [1] * 10
    #w2 = [1] * 10
    #b = [1] * 10
    #b_output = 1
    
    #w11 = [1]*5
    #w12 = [1]*5
    #w21 = [1]*5
    #w22 = [1]*5
    #wo1 = [1]*2
    #b1 = [1]*5
    #b2 = [1]*5
    #bo = 1
    
    for i in range(0, 10):
        wo[i] = random.normalvariate(0,1)
        w1[i] = random.normalvariate(0,1)
        w2[i] = random.normalvariate(0,1)
        b[i] = random.normalvariate(0,1)
    b_output = random.normalvariate(0,1)
    
    w11 = np.zeros(5)
    w12 = np.zeros(5)
    w21 = np.zeros(5)
    w22 = np.zeros(5)
    wo1 = np.zeros(2)
    b1 = np.zeros(5)
    b2 = np.zeros(5)
    bo = 0
    
    for i in range(0, 5):
        w11[i] = random.normalvariate(0,1)
        w12[i] = random.normalvariate(0,1)
        w21[i] = random.normalvariate(0,1)
        w22[i] = random.normalvariate(0,1)
        b1[i] = random.normalvariate(0,1)
        b2[i] = random.normalvariate(0,1)
    
    wo1[0] = random.normalvariate(0,1)
    wo1[1] = random.normalvariate(0,1)
    bo = random.normalvariate(0,1)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    
    x = np.arange(-5, 5.1, 0.1)
    y = np.arange(-5, 5.1, 0.1)
    xx, yy = np.meshgrid(x, y)
    
    z1 = list()
    for i in range(0,len(x)):
        for j in range(0, len(x)):
            z1.append(q1(x[i], y[j], w1, w2, wo, b, b_output))
    z1 = np.array(z1)
    z1 = z1.reshape(len(x), len(x))
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, z1,cmap = 'viridis')
    plt.show()
    
    z2 = list()
    for i in range(0,len(x)):
        for j in range(0, len(x)):
            z2.append(q3(x[i], y[j], w11, w12, w21, w22, wo1, b1, b2, bo))
    z2 = np.array(z2)
    z2 = z2.reshape(len(x), len(x))
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, z2, cmap='magma')
    plt.show()
    


    
    #y = q1(x1,x2, w1, w2, wo, b, b_output)
    #y = q3(x1,x2, w11, w12, w21, w22,wo1,b1, b2, bo)
    
