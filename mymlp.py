import pandas as pd
import numpy as np

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

for epoch in range(50):
    learning_rate = (50-epoch)/50
    for index in indices:
        cur = X[index].reshape(1,2)  # 1x2
        #print("1",cur.shape)       
        sum_h = np.dot(cur, w1) + b1        # 1x2
        #print("2",sum_h.shape)       
        out_h = sigmoid(sum_h)              # 1x2
        #print("3",out_h.shape)       
        sum_o = np.dot(out_h, w2) + b2      # 1x1
        #print("4",sum_o.shape)       
        out_o = sigmoid(sum_o)              # 1x1
        #print("5",out_o.shapr)       
        
        y = Y[index]                        
        #print("6",y.shape)       
        e = out_o - y                       # 1x1
        #print("7",e.shape)       
        dif_sig1 = out_o * (1 - out_o)      #1x1
        #print("8",dif_sig1.shape)       
        grad_w2 = e * dif_sig1 * (out_h.T)    # 2x1
        #print("9",grad_w2.shape)       
        grad_b2 = e * dif_sig1              
        #print("10",grad_b2.shape)           # 1x1
        dif_sig2 = out_h * (1 - out_h)      # 1x2
        #print("11",dif_sig2.shape)      
        grad_b1 = grad_b2 * (w2.T * dif_sig2) #2x1 
        #print("12",grad_b1.shape)       
        grad_w1 = np.dot((cur.T), grad_b1)    # 2x2
        ##print("13",grad_w1.shape)      
        
        w1 = w1 - learning_rate*grad_w1
        w2 = w2 - learning_rate*grad_w2
        b1 = b1 - learning_rate*grad_b1
        b2 = b2 - learning_rate*grad_b2
        #print(b2)

    indices = np.random.permutation(len(X)) 

headers = ["x1","x2","y"]
df = pd.read_csv("Tst.csv", names= headers)
X_test = df.drop("y", axis=1).to_numpy()
Y_test = df["y"].to_numpy()

grid = pd.read_csv("Grid.csv").to_numpy()

pr = predict(X_test).flatten()
pr[pr < 0.5] = 0
pr[pr != 0] = 1  
print("Test accuracy: ", np.mean(pr==Y_test)*100, "%")
grid_pr = predict(grid).flatten()

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

col1 = np.where(pr == 1, 'r', 'b')
col2 = np.where(grid_pr < 0.5, 'midnightblue', np.where(grid_pr < 0.5, 'teal', 
        np.where(grid_pr < 0.75, 'yellowgreen', 'yellow')))
plt.scatter(grid.T[0], grid.T[1], c = col2)
plt.scatter(X_test.T[0], X_test.T[1], c = col1)
plt.show()
