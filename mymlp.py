import numpy as np
import pandas as pd

class Layer:
    
    #A building block. Each layer is capable of performing two things:    #- Process input to get output:           output = layer.forward(input)
    
    #- Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)
    
    #Some layers also have learnable parameters which they update during layer.backward.
    
    def __init__(self):
        # Here we can initialize layer parameters (if any) and auxiliary stuff.
        # A dummy layer does nothing
        pass
    
    def forward(self, input):
        # Takes input data of shape [batch, input_units], returns output data [batch, output_units]
        
        # A dummy layer just returns whatever it gets as input.
        return input    
    
    def backward(self, input, grad_output):
        # Performs a backpropagation step through the layer, with respect to the given input.
        
        # To compute loss gradients w.r.t input, we need to apply chain rule (backprop):
        
        # d loss / d x  = (d loss / d layer) * (d layer / d x)
        
        # Luckily, we already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.
        
        # If our layer has parameters (e.g. dense layer), we also need to update them here using d loss / d layer
        
        # The gradient of a dummy layer is precisely grad_output, but we'll write it more explicitly
        num_units = input.shape[1]
        
        d_layer_d_input = np.eye(num_units)
        
        return np.dot(grad_output, d_layer_d_input) # chain rule

class Sigmoid(Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        sigmoid_forward = 1 / (1.0 + np.exp(-input))
        return sigmoid_forward
    
    def backward(self, output, grad_output):
        sigmoid_backward = output * (1.0 - output)
        result = grad_output*sigmoid_backward
        return result 

class Hidden(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_units,output_units)
        self.offset = np.random.rand(output_units)

    def forward(self,input):
        h_forward = np.dot(input,self.weights) + self.offset
        return h_forward
    
    def backward(self,input, error):
        grad_input = np.dot(error, self.weights.T)
        
        grad_weights = np.dot(input.T, error)
        self.weights = self.weights - self.learning_rate * grad_weights
        self.offset = self.offset - self.learning_rate * error 
        
        return grad_input

headers = ["x1", "x2", "y"]
df = pd.read_csv("Trn.csv", names = headers)

X_train  = df.drop(["y"], axis=1).to_numpy(dtype = np.float128)[:40]
Y_train  = df["y"].to_numpy(dtype=int)[:40]

mlp = []
mlp.append(Hidden(2, 3, learning_rate=1))
mlp.append(Sigmoid())
mlp.append(Hidden(3, 1, learning_rate=1))
mlp.append(Sigmoid())

def forward(mlp, X):
    
    activations = []
    input = X
    for l in mlp:
        activations.append(l.forward(input))
        input = activations[-1]
        
    assert len(activations) == len(mlp)
    return activations

def classify(Y):  
    Y[Y < 0.5] = 0
    Y[Y != 0] = 1
    return Y
    
def train(network,X,y,index):
    layer_activations = forward(network,X)
    layer_inputs = [X] + layer_activations
    h_i_3 = layer_activations[-1]

    error = h_i_3[index] - y[index]
    
    for layer_index in reversed(range(len(network))):
        layer = network[layer_index]
        error = layer.backward(layer_inputs[layer_index],error)
        
train_log = []

indices = np.random.permutation(len(X_train))

for epoch in range(25):    
    for index in indices:
        train(mlp,X_train,Y_train, index)
    
    pr = forward(mlp,X_train)[-1]
    pr = classify(pr)
    train_log.append(np.mean(pr==Y_train))
    
    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
