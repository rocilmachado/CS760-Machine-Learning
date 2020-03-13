'''
Created on Nov 25, 2019

@author: Rocil
'''
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from numpy import empty
from sklearn.utils.arrayfuncs import min_pos


def signum(x):
    if (x >= 0):
        return 1
    else:
        return 0


def TrueError (low_limit, high_limit,num):
    #Generate random floating point numbers in [low_limit, high_limit] range
    x = []
    y = []
    for i in range (0,num):
        randnum = random.uniform(low_limit, high_limit)
        x.append( randnum)
        y.append(signum(randnum))
    #print("Random numbers:", x)
    x.sort()
    min_pos = 0
    positive_value_found = 0
    for i in range(len(x)):
        if(x[i] >= 0):
            min_pos = x[i]
            positive_value_found = 1
            break
    #print("index at break:", i)
    if(positive_value_found == 0):
        min_pos = 1
    
    #true_error = min_pos
    #print("Minimum positive number:", min_pos)
    return min_pos/2
     

if __name__ == "__main__":
    
    true_errors = []
    for i in range(0,10000):
        true_errors.append(TrueError(-1,1,200))
    
    #print("True errors:", len(true_errors))
    #print("True_error_min:", min(true_errors))
    #print("True_error_max:", max(true_errors))
    
    #plt.hist(true_errors, bins=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
    plt.hist(true_errors, bins='auto')
    #plt.hist(true_errors, bins = 10)
    plt.xlabel('Risk')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()
    