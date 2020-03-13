import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def reLU(x):
    return max(0,x)

def reLU_derivative(x):
    if( x >= 0):
        return 1
    else:
        return 0

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.inputA = [1,x[0],x[1]]
        self.inputB = [1,x[0],x[1]]
        #self.weights1   = np.random.rand(self.input.shape[1],4) 
        #self.weights2   = np.random.rand(4,1)
        
        #self.weights1 = np.array([0.1, 0.2,  0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        #self.weights1 = np.array([1, 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2])
        #self.weights1 = np.array([4,3,2,1,0,-1,-2,-3,-4])
        self.weights1 = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9])            
        self.y          = y
        #self.output     = np.zeros(self.y.shape)

    def feedforward(self,x,y):
        self.weightsA = self.weights1[0:3]
        self.weightsB = self.weights1[3:6] 
        self.weightsC = self.weights1[6:9]
        self.x = x
        self.y = y
        self.inputA = [1,self.x[0],self.x[1]]
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
        
    def backpropagation(self):
        self.E = (1/2) * (pow((self.output - self.y),2))
        #print("Old E: ", E, "Output: ", self.output)
        delta_E_vC = self.output - self.y
        delta_E_uC = delta_E_vC * sigmoid_derivative(self.u_C)
        #print("Question2: E:", self.E, "delta_E_vC", delta_E_vC, "delta_E_uC", delta_E_uC)
        delta_E_vA = self.weights1[7] * delta_E_uC
        delta_E_uA = delta_E_vA * reLU_derivative(self.u_A)
        delta_E_vB = self.weights1[8] * delta_E_uC
        delta_E_uB = delta_E_vB * reLU_derivative(self.u_B)
        #print("Question3: delta_E_vA: ", delta_E_vA, " delta_E_uA :", delta_E_uA, "delta_E_vB: ", delta_E_vB, " delta_E_uB :", delta_E_uB)
        delta_E_u = [delta_E_uA, delta_E_uB , delta_E_uC]
        self.delta_E_w = np.zeros(9)
        index = 0
        for i in range (0, 3):
            for j in range (0,3):
                if(i<2):
                    self.delta_E_w[index] = delta_E_u[i] * self.inputA[j]
                else:
                    self.delta_E_w[index] = delta_E_u[i] * self.inputC[j]
                index = index + 1
        #print("Question4: delta_E_w : ", self.delta_E_w)
        
            
    def gradientdescent(self):
        for i in range (0, len(self.weights1)):
            self.weights1[i]= self.weights1[i] - 0.1 * self.delta_E_w[i]
        #print("Updated Weights:", self.weights1)
        
    def predictoutput(self, x):
        input_x = [1,x[0],x[1]]
        weightsA = self.weights1[0:3]
        weightsB = self.weights1[3:6] 
        weightsC = self.weights1[6:9]
        #output_y = np.zeros(1)
        u_A = np.dot(input_x, weightsA)
        v_A = reLU(u_A)
        u_B = np.dot(input_x, weightsB)
        v_B = reLU(u_B)
        inputC = [1, v_A, v_B]
        u_C = np.dot(inputC, weightsC)
        output_y = sigmoid(u_C)
        v_C = output_y
        #print("Question1: Weights: ", self.weights1, "Input:", input_x, u_A, " ", v_A," ", u_B, " ", v_B, " ", u_C, " ", v_C)
        return output_y
        
    def calculateloss(self):
        E = (1/2) * (pow((self.output - self.y),2))
        #print("E after updating: ", E)
        
    def trainingseterror(self, y_actual, y_pred):
        E = (1/2) * (pow((y_actual - y_pred),2))
        return E
        
                 
    #def backprop(self):
        ## application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        #d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        #d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        #self.weights1 += d_weights1
        #self.weights2 += d_weights2


if __name__ == "__main__":
    balance_data = pd.read_csv( 
        r'C:\Users\Rocil\Documents\Machine Learning\HW6\data.txt',
#'http://pages.cs.wisc.edu/~jerryzhu/cs760/hw2/D1.txt', 
    sep= ' ', header = None)
    
    #balance_data = np.random.shuffle(balance_data[:,:])
    
    
    #X = balance_data.values[random_i,0:2]
    #y = balance_data.values[random_i:,2]
    #print(X)
    
    
    #X = np.array([1,-1])
    #X = np.array([[-0.2,1.7]])
    #X = np.array([[-4,1]])
    
    #y = np.array([1])
    #y = np.array([[0]])
    
    
    #Questions1-5
    #nn.feedforward()
    #nn.backpropagation()
    #nn.gradientdescent()
    #nn.feedforward()
    #nn.calculateloss()
    error_array = np.zeros(100)
    iteration = 0;
    print("Size of data : ", balance_data.shape[0])
    random_i = np.random.choice(balance_data.shape[0])
    #print(random_i)
    X = balance_data.values[random_i,0:2]
    #X = balance_data.values[iteration,0:2]
    y = balance_data.values[random_i,2]
    #y = balance_data.values[iteration,2]
    nn = NeuralNetwork(X,y)
    error_index = 0
    

    while iteration < 10000:
        nn.feedforward(X,y)
        nn.backpropagation()
        nn.calculateloss()
        nn.gradientdescent()
        iteration = iteration +1
        if (iteration % 100 == 0):
            totalError = 0
            for k in range(0,balance_data.shape[0]):
                y_pred = nn.predictoutput(balance_data.values[k,0:2])
                totalError = totalError + nn.trainingseterror(balance_data.values[k,2], y_pred)
                #print("Actual : ", balance_data.values[k,2], "Predicted : ", y_pred)
            error_array[error_index] = totalError
            print(totalError)
            error_index = error_index + 1
        random_i = np.random.choice(balance_data.shape[0])
        #print(random_i)
        #X = balance_data.values[iteration,0:2]
        #y = balance_data.values[iteration,2]
        
        X = balance_data.values[random_i,0:2]
        y = balance_data.values[random_i,2] 
        
    plt.plot(error_array)
    # naming the x axis 
    plt.xlabel('Number of Rounds') 
# naming the y axis 
    plt.ylabel('Training Set Error') 
  
# giving a title to my graph 
    plt.title('Plot of Training Set Error vs Number of Rounds')    
    plt.show()
    
    print(error_array)        
    print(nn.output)
