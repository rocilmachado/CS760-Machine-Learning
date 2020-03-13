'''
Created on Sep 20, 2019

@author: Rocil Machado
'''
# Run this program on your local python 
# interpreter, provided you have installed 
# the required libraries. 

# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix
import sklearn.datasets as datasets
#from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.datasets.base import load_data
import pydotplus
from sklearn.feature_selection.tests.test_base import feature_names
from Tools.demo.sortvisu import randomize

global randomize_flag
randomize_flag = 1

# Function importing Dataset 
def importdata(): 
    balance_data = pd.read_csv( 
        #'C:/Users/Rocil/Documents/Machine Learning/test.txt',
'http://pages.cs.wisc.edu/~jerryzhu/cs760/hw2/D1.txt', 
    sep= ' ', header = None) 
    
    # Printing the dataswet shape 
    print ("Dataset Length: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
    
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 

# Function to split the dataset 
def splitdataset(balance_data):
    
    # Seperating the target variable 
    X = balance_data.values[:, 0:2]
    Y = balance_data.values[:, 2]
    
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, Y, test_size = 0.3, random_state = 100) 
    
    return X, Y, X_train, X_test, y_train, y_test

"""
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 

    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 

    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
"""    
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
    
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 30, min_samples_leaf = 1, presort= False)
    

    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    
    export_graphviz(clf_entropy, out_file='tree.dot', class_names=["negative", "positive"],  
                filled=True, rounded=True,
                special_characters=True)

    from subprocess import call
    from IPython.display import Image
    Image(filename = 'tree.png')
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=500'])
    
    import matplotlib.pyplot as plt
    plt.figure(figsize = (120, 120))
    plt.imshow(plt.imread('tree.png'))
    plt.axis('off');
    plt.show();

    return clf_entropy


# Function to make predictions 
def prediction(X_test, clf_object): 

    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
    
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
    
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
    
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
    
    print("Report : ", 
    classification_report(y_test, y_pred)) 

# Driver code 
def main():
    
    # Building Phase 
    data = importdata() 
    X, Y, X_train, y_train, X_test, y_test = splitdataset(data)
    #X, Y, X_32, X_128, X_512,X_2048, X_8192, Y_32,Y_128, Y_512,Y_2048, Y_8192,X_test, y_test = splitdataset(data) 
    #clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X, X_test, Y)
    #clf_entropy = tarin_using_entropy(X, X_test, Y)    ## Do not split the training set into training and test


    # Operational Phase 
    #print("Results Using Gini Index:") 
    
    # Prediction using gini 
    #y_pred_gini = prediction(X_test, clf_gini) 
    #cal_accuracy(y_test, y_pred_gini) 
    
    print("Results Using Entropy:") 
    # Prediction using entropy 
    
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    
# Calling main function 
if __name__=="__main__": 
    main() 
