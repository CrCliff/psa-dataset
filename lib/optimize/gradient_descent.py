import numpy as np
import sys

class GradientDescent():
    
    LEARNING_RATE=1e-2
    
    def __init__(self, *, data, weights, loss, gradient, learning_rate=LEARNING_RATE):
        self.data = data
        self.weights = weights
        self.loss = loss
        self.gradient = gradient
        
        self.learning_rate = learning_rate
        
        self._hist_weights = [weights]
        self._hist_loss = [self.eval()]
    
    def minimize(self, epoch_size=1):
        for epoch in range(epoch_size):
            try:
                self.update()
                self._hist_weights.append(self.weights)
                self._hist_loss.append(self.eval())
            except FloatingPointError:
                print("Overflow or underflow has occured! Halting...")
                break
    
    def hist_weights(self):
        return np.array(self._hist_weights)
    
    def hist_loss(self):
        return np.array(self._hist_loss).T
            
    def eval(self):
        return self.loss(self.data, self.weights)
        
    def update(self):
        G = self.gradient(self.data, self.weights)
        self.weights = self.weights - self.learning_rate * G