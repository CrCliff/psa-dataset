import numpy as np
import math

class UnflattenLayer():
    
    def __init__(self, shape):
        self.shape = shape
    
    def forward_propagate(self, X):
        self.X = X
        
        return X.reshape(self.shape)
    
    def gradient(self, prev_grad):
        return prev_grad.flatten()
    
    def is_elementwise(self):
        return False
