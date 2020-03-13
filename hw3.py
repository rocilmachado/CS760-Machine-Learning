'''
Created on Oct 1, 2019

@author: Rocil
'''
import pandas as pd
import scipy as sp
import numpy as np
import scipy.linalg as linearalg

#filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv'
#df = pd.read_csv(filepath).iloc[:, [0,4,6]]
#df.head()

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = linearalg.inv(cov)
    print("X-u:", x_minus_mu)
    print("cov", cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


def createtestset():
    testset = pd.DataFrame()
    i = 1
    j = 1
    for m in np.arange (-2, 0.1, 2):
        for n in np.arange (-2, 0.1, 2):
            testset.values[i][j] = [m,n]
            print("testset:", testset[i])
            i = i+1
    return testset

#df_x = df[['carat', 'depth', 'price']].head(500)
#df_x['mahala'] = mahalanobis(x=df_x, data=df[['carat', 'depth', 'price']])
#df_x.head()


df = pd.read_csv( 
        'C:/Users/Rocil/Documents/Machine Learning/HW3/D2a.txt',
    sep= ' ',header = None)

X = df.values[:,0:6]
Y = df.values[:,6] 

#xtest = pd.read_csv( 
        #'C:/Users/Rocil/Documents/Machine Learning/HW3/testset.csv',
#'http://pages.cs.wisc.edu/~jerryzhu/cs760/hw2/D1.txt', 
    #sep=',',header = None)  
                   

#df.dropna(inplace=True)  # drop missing values.
df.head()

from sklearn.model_selection import train_test_split
#xtrain, xtest, ytrain, ytest = train_test_split(df.drop('Class', axis=1), df['Class'], test_size=.3, random_state=100)
xtrain, xtest, ytrain, ytest = train_test_split( X, Y, test_size = 0.3, random_state = 100)

X= df.drop('Class', axis=1)
Y=df['Class']
print("X train:", xtrain)
print("Y train:", ytrain)

# Split the training data as pos and neg
xtrain_pos = xtrain.loc[ytrain == 1, :]
xtrain_neg = xtrain.loc[ytrain == 0, :]

xtrain_pos = X.loc[Y==1,:]
xtrain_neg = X.loc[Y==0,:]

print("X train positive:", xtrain_pos)
print("X train nagative:", xtrain_neg)

print("X test:", xtest)

class MahalanobisBinaryClassifier():
    def __init__(self, X, Y):
        self.xtrain_pos = xtrain.loc[Y == 1, :]
        self.xtrain_neg = xtrain.loc[Y == 0, :]

    def predict_proba(self, xtest):
        pos_neg_dists = [(p,n) for p, n in zip(mahalanobis(xtest, self.xtrain_pos), mahalanobis(xtest, self.xtrain_neg))]
        return np.array([(1-n/(p+n), 1-p/(p+n)) for p,n in pos_neg_dists])

    def predict(self, xtest):
        return np.array([np.argmax(row) for row in self.predict_proba(xtest)])


clf = MahalanobisBinaryClassifier(xtrain, ytrain)        
pred_probs = clf.predict_proba(xtest)
pred_class = clf.predict(xtest)

# Pred and Truth
pred_actuals = pd.DataFrame([(pred, act) for pred, act in zip(pred_class, ytest)], columns=['pred', 'true'])
print(pred_actuals[:5])


from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
truth = pred_actuals.loc[:, 'true']
pred = pred_actuals.loc[:, 'pred']
scores = np.array(pred_probs)[:, 1]
print('AUROC: ', roc_auc_score(truth, scores))
print('\nConfusion Matrix: \n', confusion_matrix(truth, pred))
print('\nAccuracy Score: ', accuracy_score(truth, pred))
print('\nClassification Report: \n', classification_report(truth, pred))