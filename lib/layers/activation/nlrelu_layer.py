import numpy as np
from .activation_layer import ActivationLayer

class NLReluLayer(ActivationLayer):
    
    def __init__(self, alpha=1, beta=0.5, leaky=False):
        self.alpha = alpha
        self.beta = beta
        self.leaky = leaky
    
    def forward_propagate(self, X, otypes=[np.float]):
        self.X = X
        return np.vectorize(self.g, otypes)(X)
    
    def g(self, x):
        return (
            self.alpha * np.log(self.beta * x + 1.)
            if x >= 0
            else (0.1*x if self.leaky else 0.)
        )
    
    def gradient(self, otypes=[np.float]):
        def grad_z(z):
            return (
                self.alpha * self.beta / (self.beta * z + 1.)
                if z >= 0
                else (0.1 if self.leaky else 0.)
            )
        
        return np.vectorize(grad_z, otypes)(self.X)