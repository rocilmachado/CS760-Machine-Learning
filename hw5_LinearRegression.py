'''
Created on Oct 15, 2019

@author: Rocil
'''
import numpy as np
import pandas as pd
from cmath import sqrt
from builtins import len
from numpy import empty, gradient
from numpy.random.mtrand import beta


def importdata(filename): 
    balance_data = pd.read_csv(filename, sep= ',', header = None)
    
    # Printing the dataswet shape 
    print ("Dataset Length: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
    
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    #balance_data.to_numpy() 
    return balance_data

def splitdataset(balance_data):
    
    # Seperating the target variable 
    X = balance_data.values[:, 0:1]
    Y = balance_data.values[:, 1]
    #print("Y:",Y)
    X_train = X[0:116]
    Y_train = Y[0:116]
    #print("Y_train:",Y_train)
    X_test = X[116:164]
    Y_test = Y[116:164] 
    
    return X, Y, X_train, X_test, Y_train, Y_test

def normalize(feature):
    #print("Feature:", feature)
    sigma = 0
    mean = sum(feature) / len(feature)
    for i in feature:
        #print("Feature[i]:",i)
        sigma = sigma + pow((i- mean),2)
    sigma = sigma/(len(feature)-1)
    sigma  = sqrt(sigma)
    #print("sigma:", sigma)
    #print("Feature after normalising:", feature)
    #print("Mean of the data:", mean)
    #print("Standard devaition of the data:", sigma)
    return mean, sigma

def CreateDesignMatrix(X,Y, n):
    X_des = empty([n,3])
    X_des[:,0] = 1
    for i in range (0,len(X)):
        X_des[i,1] = X[i]
        X_des[i,2] = Y[i]
    print("Design Matrix:", X_des)
    return X_des

def CreateDesignMatrixQ8(X, n):
    X_des = empty([n,2])
    X_des[:,0] = 1
    for i in range (0,len(X)):
        X_des[i,1] = X[i]
    #print("Design Matrix:", X_des)
    return X_des
    
def Predict(X_test, Y_test, beta):
    X = CreateDesignMatrix(X_test, Y_test, len(X_test))
    y_pred = X.dot(beta)
    #for i in range (0, len(X_test)):
        #y_pred = beta[0] + beta[1]* X_test[i] + beta[2] * Y_test 
    return y_pred

def PredictQ8(X_test, gamma):
    X = CreateDesignMatrixQ8(X_test, len(X_test))
    y_pred = X.dot(gamma)
    #for i in range (0, len(X_test)):
        #y_pred = beta[0] + beta[1]* X_test[i] + beta[2] * Y_test 
    return y_pred

def CalculateError(y_pred, y_actual):
    mean = sum(y_actual) / len(y_actual)
    print("Mean:", mean)
    mean_squares = 0
    R_sum = 0
    for i in range (0, len(y_pred)):
        mean_squares = mean_squares + pow((y_pred[i]- y_actual[i]),2)
    SS_res = mean_squares
    print("SS_Res: ", SS_res)
    for i in range (0, len(y_actual)):
        print("Actual y:", y_actual[i])
        R_sum = R_sum + pow((y_actual[i]- mean),2)
    SS_tot = R_sum
    print("SS_tot: ", SS_tot)
    R_square_error = 1 - (SS_res/SS_tot)  
    mean_squared_error = mean_squares/(len(y_pred)) 
    return mean_squared_error, R_square_error


def CalculateGradient(XiT,Y):
    #XiT = Design Matrix = nx3 dimensions
    #beta = 3X1 dimensions
    #Y = Training values
    difference = np.empty([len(Y), 1])
    beta = np.empty([3,1])
    product = XiT.dot(beta)
    #print("Product:", product.shape)
    print("Product dimensions:",product.shape)
    print("Y dimensions:", Y.shape)
    for i in range(0, len(Y)):
        difference[i] = product[i] - Y[i]
    difference = difference.transpose()
    term = difference.dot(XiT)
    gradient = term * 2/len(Y)
    #print("Term:", term)
    #print("Gradient :", gradient)
    return gradient

def TrainBeta(gradient, theta):
    print("Gradient :", gradient[0]) 
    beta_values = np.empty([10,3])
    beta_values[0] = 0
    print("Beta values :", beta_values[0])
    for i in range(1,10):
        beta_values[i] = beta_values[i-1] - (theta)*gradient
        print("Beta values for iteration ", i ," : ", beta_values[i] )
    return beta_values

def main():
    filename = r'C:\Users\Rocil\Documents\Machine Learning\HW5\mendota.txt'
    data_mendota = importdata(filename) 
    X_mendota, Y_mendota, X_train_mendota, X_test_mendota, Y_train_mendota, Y_test_mendota = splitdataset(data_mendota)
    #print("Mendota Training:", Y_train_mendota)
    filename = r'C:\Users\Rocil\Documents\Machine Learning\HW5\monona.txt'
    data_monona = importdata(filename) 
    X_monona, Y_monona, X_train_monona,X_test_monona, Y_train_monona, Y_test_monona = splitdataset(data_monona)
    #print("Monona Training:", Y_train_monona)
    mean_mendota_training, std_mendota_training = normalize(Y_train_mendota)
    mean_monona_training, std_monona_training = normalize(Y_train_monona)
    print("Mean of the training data - Mendota:", mean_mendota_training)
    print("Std deviation of the training data - Mendota:", std_mendota_training )
    print("Mean of the training data - Monona:", mean_monona_training)
    print("Std deviation of the training data - Monona:", std_monona_training )
    X = CreateDesignMatrix(X_train_mendota, Y_train_monona, len(X_train_mendota))
    XT = X.transpose()
    print("Y_train_Mendota:", Y_train_mendota)
    beta1 = np.linalg.inv(XT.dot(X))
    beta2 = XT.dot(Y_train_mendota) 
    beta = beta1.dot(beta2)
    print("Beta: ", beta)
    Y_pred = Predict(X_test_mendota, Y_test_monona, beta)
    print("Y_predicted for Mendota:", Y_pred)
    mean_squared_error, R_square_error = CalculateError(Y_pred, Y_test_mendota)
    print("Mean Squared Error:", mean_squared_error,", R Squared error:", R_square_error)
    
    
    X1 = CreateDesignMatrixQ8(X_train_mendota, len(X_train_mendota))
    X1T = X1.transpose()
    gamma1 = np.linalg.inv(X1T.dot(X1))
    gamma2 = X1T.dot(Y_train_mendota)
    gamma = gamma1.dot(gamma2)
    print("Gamma: ", gamma)
    Y_pred_q8 = PredictQ8(X_test_mendota, gamma)
    print("Y_predicted for Mendota without Y monona:", Y_pred_q8)
    
    gradient = CalculateGradient(X, Y_train_mendota)
    print("Gradient :", gradient)
    
    beta_values = TrainBeta(gradient, 0.1)
    print("Beta values:", beta_values)
    
# Calling main function 
if __name__=="__main__": 
    main()