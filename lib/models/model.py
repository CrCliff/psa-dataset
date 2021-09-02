import numpy as np
from overrides import EnforceOverrides
from lib.layers.conv import ConvolutionalLayer, FlattenLayer, UnflattenLayer, PoolingLayer

class Model(EnforceOverrides):
    '''
    A simple pipeline for composing layers to create a network.
    During forward propagation, the output of each layer is used as the input for the next layer.
    Each layer must implementation the forward_propagate(X) method.
    During back propagation, the gradient of each layer is used to compute the gradient of the previous layer.
    Each layer must implement the gradient() method.
    '''
    
    LEARNING_RATE=1e-4
    
    def __init__(self, layers=[], verbose=False):
        self.layers = layers
        self.verbose = verbose
        
    def train(self, X, y, batch_size=0):
        if batch_size <= 0:
            # Don't batch
            self.forward_propagate(X)
            self.back_propagate(y)
        else:
            # Use batch size
            N = X.shape[0]
            for i in range(0,N,batch_size):
                start = i
                end = min(i+batch_size,N)
                
                X_batch = X[start:end:]
                y_batch = y[start:end:]
            
                self.forward_propagate(X_batch)
                self.back_propagate(y_batch)
        
    def forward_propagate(self, X):
        result = X
        
        i = 0
        for layer in self.layers:
            result = layer.forward_propagate(result)
            i += 1
            if self.verbose:
                print(f'Output of layer {i}:\n', result[:5])
        
        return result
    
    def back_propagate(self, y):
        output = self.layers[-1]
        grad = output.gradient(y)
        
        for layer in self.layers[-2:0:-1]:
            if layer.is_elementwise():
                grad = grad * layer.gradient()
            else:
                if (hasattr(layer, 'weights')
                       or isinstance(layer, ConvolutionalLayer)):
                    # Update weights using prev gradient
                    layer.update(grad)
                    
                if (isinstance(layer, ConvolutionalLayer)
                        or isinstance(layer, FlattenLayer)
                        or isinstance(layer, PoolingLayer)
                        or isinstance(layer, UnflattenLayer)):
                    grad = layer.gradient(grad)
                else:    
                    grad = grad @ layer.gradient()
        
        return grad
    
    def eval(self, X):
        return self.forward_propagate(X)
    
    def loss(self, X, y_act):
        self.eval(X)
       
        return self.layers[-1].eval(y_act)