from numpy import empty
import pandas as pd
import numpy as np
from cmath import sqrt
from builtins import len

filename = r'C:/Users/Prajna/Desktop/mendota_clean.csv'
balance_data = pd.read_csv(filename, sep= ',', header = None)
X_mendota = balance_data.values[:, 0:1]
Y_train_mendota = balance_data.values[0:116, 1]
X_test_mendota = balance_data.values[116:,0]
Y_test_mendota = balance_data.values[116:,1]


filename = r'C:/Users/Prajna/Desktop/monona_clean.csv'
balance_data = pd.read_csv(filename, sep= ',', header = None)
Y_train_feature_monona = balance_data.values[0:116, 1]
Y_test_feature_monona = balance_data.values[116:, 1]

X_train_final = empty([116,3])
X_train_final[:,0] = 1
for i in range (0,len(X_train_final)):
    X_train_final[i,1] = X_mendota[i]
    X_train_final[i,2] = Y_train_feature_monona[i]
    #print("Design Matrix:", X_des)
    
X_test_final = empty([48,3])
X_test_final[:,0] = 1
for i in range (0,len(X_test_final)):
    X_test_final[i,1] = X_test_mendota[i]
    X_test_final[i,2] = Y_test_feature_monona[i]
#print('SHAPE OF XTEST IS', np.shape(X_Test))

Y_train_mendota_array = empty([116,1])
for i in range (0,len(Y_train_mendota_array)):
    Y_train_mendota_array[i] = Y_train_mendota[i]
YT=Y_train_mendota_array.transpose()

beta =  np.zeros([1,3])
betaf =  np.zeros([1,3])
num = 0.0000002/116
#print(num)


for i in range(1000000):
    print('Beta Transpose is',betaf)
    betafT=betaf.transpose()
    #print('SHAPE OF betafT', np.shape(betafT))
    #print('SHAPE OF betaf', np.shape(betaf))
    #print('SHAPE OF X_test_final', np.shape(X_test_final))
    #y_pred = X_test_final.dot(betafT)
    y_pred = np.matmul(X_train_final,betafT)
    #y_check= YT.dot(X_train_final)
    #print('SHAPE OF Y_PRED', np.shape(y_pred))
    
    #ERRORS
    mean_squares = 0
    mse = 0
    for j in range (0, len(y_pred)):
        #mean_squares = mean_squares + pow((y_pred[j]- Y_test_mendota[j]),2)
        mean_squares = mean_squares + pow((y_pred[j]- Y_train_mendota[j]),2)
    #print('SHAPE of mean squares', np.shape(mean_squares))
    mse = mean_squares/116
    print(mse)
    
    #next beta calculation
    yp = np.zeros([116,1])
    yp = np.matmul(X_train_final,betafT)
    ypT = yp.transpose()
    #print('SHAPE OF X_train_final', np.shape(X_train_final))
    
    #print('SHAPE OF ypT', np.shape(ypT))
    diff = np.zeros([116,1])
    diff = np.subtract(yp, Y_train_mendota_array)
    #print('SHAPE OF YT', np.shape(YT))
    #print('SHAPE OF diff', np.shape(diff))
    diffT = diff.transpose()
    diffX = np.matmul(diffT,X_train_final)
    #print('Gradient is', diffX)
    #print('SHAPE OF diffX', np.shape(diffX))
    diffxnum = diffX * num
    betaf = np.subtract(betaf, diffxnum)
    #print('SHAPE OF BETA', np.shape(betaf))
