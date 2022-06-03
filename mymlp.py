import pandas as pd
import numpy as np
import time
startTime = time.time()
headers = ["x1","x2","y"]
df = pd.read_csv("Trn.csv", names= headers)
X = df.drop("y", axis=1).to_numpy()
Y = df["y"].to_numpy()

w1 = np.random.rand(2, 5) #first matrice of weights
b1 = np.random.rand(5).reshape(1,5)    #offset1
w2 = np.random.rand(5, 1) #second matrice of weights
b2 = np.random.rand(1).reshape(1,1)    #offset2
learning_rate = 0.1


def sigmoid(input):
    return 1/(1+np.exp(-input))

def predict(input):
    sum_h = np.dot(input, w1) + b1        # 1x2
    out_h = sigmoid(sum_h)              # 1x2
    sum_o = np.dot(out_h, w2) + b2      # 1x1
    result = sigmoid(sum_o)              # 1x1
    
    return result

indices = np.random.permutation(len(X))

for epoch in range(200):
    for index in indices:
        cur = X[index].reshape(1,2)  # 1x2
        sum_h = np.dot(cur, w1) + b1        # 1x2
        out_h = sigmoid(sum_h)              # 1x2
        sum_o = np.dot(out_h, w2) + b2      # 1x1
        out_o = sigmoid(sum_o)              # 1x1
        
        y = Y[index]                        
        e = out_o - y                       # 1x1
        dif_sig1 = out_o * (1 - out_o)      #1x1
        grad_w2 = e * dif_sig1 * (out_h.T)    # 2x1
        grad_b2 = e * dif_sig1              
        dif_sig2 = out_h * (1 - out_h)      # 1x2
        grad_b1 = grad_b2 * (w2.T * dif_sig2) #2x1 
        grad_w1 = np.dot((cur.T), grad_b1)    # 2x2
        
        w1 = w1 - learning_rate*grad_w1
        w2 = w2 - learning_rate*grad_w2
        b1 = b1 - learning_rate*grad_b1
        b2 = b2 - learning_rate*grad_b2
        
    indices = np.random.permutation(len(X)) 

executionTime = (time.time() - startTime)
headers = ["x1","x2","y"]
df = pd.read_csv("Tst.csv", names= headers)
X_test = df.drop("y", axis=1).to_numpy()
Y_test = df["y"].to_numpy()

grid = pd.read_csv("Grid.csv").to_numpy()


pr = predict(X).flatten()
pr[pr < 0.5] = 0
pr[pr != 0] = 1  
print('Training time in seconds: ' + str(executionTime))
print("Train accuracy: ", np.mean(pr==Y)*100, "%")

pr = predict(X_test).flatten()
pr[pr < 0.5] = 0
pr[pr != 0] = 1  
print("Test accuracy: ", np.mean(pr==Y_test)*100, "%")
grid_pr = predict(grid).flatten()

import matplotlib.pyplot as plt

col1 = np.where(Y_test == 1, 'r', 'b')
col2 = np.where(grid_pr < 0.5, 'midnightblue', np.where(grid_pr < 0.5, 'teal', 
        np.where(grid_pr < 0.75, 'yellowgreen', 'yellow')))
plt.scatter(grid.T[0], grid.T[1], c = col2)
plt.scatter(X_test.T[0], X_test.T[1], c = col1)
plt.show()
