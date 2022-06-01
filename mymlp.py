import numpy as np
import pandas as pd

class Hidden:
    def __init__(self, input_units, output_units, learning_rate=0.1):
        
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_units,output_units)
        self.offset = np.random.rand(output_units)
        
    def forward(self,input):
        
        h_forward = np.dot(input,self.weights) + self.offset
        sigmoid_forward = 1 / (1 + np.exp(-h_forward))
        #print(sigmoid_forward)
        return sigmoid_forward
    
    def backward(self,input, output, error):
        sigmoid_backward = output * (1.0 - output)
        error = np.dot(error, self.weights.T)
        delta = error * sigmoid_backward
        
        grad_weights = np.dot(input, delta)
        
        self.weights = self.weights - self.learning_rate * grad_weights
        self.offset = self.offset - self.learning_rate * error 
        
        return delta

headers = ["x1", "x2", "y"]
df = pd.read_csv("Trn.csv", names = headers)

X_train  = df.drop(["y"], axis=1).to_numpy(dtype = np.float128)[:10]
Y_train  = df["y"].to_numpy(dtype=int)

mlp = []
mlp.append(Hidden(2, 5, learning_rate=1))
mlp.append(Hidden(5, 1, learning_rate=1))

def forward(mlp, X):
    
    activations = []
    input = X
    for l in mlp:
        activations.append(l.forward(input))
        input = activations[-1]
        
    assert len(activations) == len(mlp)
    
    return activations
    
def train(network,X,y,index):
    layer_activations = forward(network,X)
    layer_inputs = [X] + layer_activations
    h_i_3 = layer_activations[-1]
    
    e_i = h_i_3[index] - y[index]

    for layer_index in reversed(range(len(network)-1)):
        layer = network[layer_index]
        e_i = layer.backward(layer_inputs[layer_index],layer_inputs[layer_index+1],e_i)
        
    return np.mean(e_i)        

train_log = []

indices = np.random.permutation(len(X_train))

for epoch in range(25):    
    for index in indices:
        train(mlp,X_train,Y_train, index)
    
    pr = (forward(mlp,X_train)[-1])
    train_log.append(np.mean(pr==Y_train))
    
    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
