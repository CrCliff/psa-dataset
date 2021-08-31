import numpy as np
import sys
from .gradient_descent import GradientDescent

class ADAM(GradientDescent):
    
    LEARNING_RATE=1e-2
    DECAY_RATE1=0.9
    DECAY_RATE2=0.999
    NUMERIC_STABILITY=1e-8
    
    def __init__(self, *, data, weights, loss, gradient, learning_rate=LEARNING_RATE, decay_rate1=DECAY_RATE1, decay_rate2=DECAY_RATE2, numeric_stability=NUMERIC_STABILITY):
        super().__init__(
            data=data,
            weights=weights,
            loss=loss,
            gradient=gradient,
            learning_rate=learning_rate
        )
        print(learning_rate)
        self.decay_rate1=decay_rate1
        self.decay_rate2=decay_rate2
        self.numeric_stability=numeric_stability
        self.mom1 = 0
        self.mom2 = 0
        
    def update(self):
        rho1 = self.decay_rate1
        rho2 = self.decay_rate2
        delta = self.numeric_stability
        G = self.gradient(self.data, self.weights)
        
        self.mom1 = rho1*self.mom1+(1-rho1)*G
        self.mom2 = rho2*self.mom2+(1-rho2)*(G*G)
        
        update_param = (
            (self.mom1 / (1 - rho1)) /
            (np.sqrt(self.mom2/(1-rho2))+delta)
        )
        
        self.weights = self.weights - self.learning_rate * update_param