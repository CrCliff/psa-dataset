import numpy as np

class FullyConnectedLayer():
    '''
    An implementation of a fully connected layer.
    The layer is initialized with random weights (W) and biases(b).
    The input dimension must match the output dimension of the previous layer.
    The result of forward propagation is X*W + b.
    '''
    
    SCALE = 1e-4
    LEARNING_RATE = 1e-4
    
    def __init__(self, dim_in, dim_out, learning_rate=LEARNING_RATE, scale=SCALE):
        self.weights = (np.random.rand(dim_in, dim_out) - 0.5) * scale
        self.biases = (np.random.rand(1, dim_out) - 0.5) * scale
        self.learning_rate = learning_rate
    
    def forward_propagate(self, X):
        self.X = X
        if len(self.X.shape) < 2:
            self.X = np.reshape(self.X, (1,-1))
        return X @ self.weights + self.biases
    
    def gradient(self):
        return self.weights.T
    
    def is_elementwise(self):
        return False
    
    def update(self, grad):
        self._update_weights(grad)
        self._update_biases(grad)
    
    def _update_weights(self, grad):
        N = self.X.shape[0]
        grad_weights = self.X.T@grad
        self.weights = self.weights - self.learning_rate * grad_weights
        
    def _update_biases(self, grad):
        N = self.X.shape[0]
        grad_biases = np.ones([N,1]).T@grad
        self.biases = self.biases - (self.learning_rate) * grad_biases
    
from unittest import TestCase

class FullyConnectedLayerTests(TestCase):
    
    @staticmethod
    def test_gradient():
        X = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        layer = FullyConnectedLayer(2, 4)
        np.testing.assert_equal(layer.gradient(), layer.weights.T)
        
    @staticmethod
    def test_dims_match():
        X = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        layer = FullyConnectedLayer(2, 4)
        np.testing.assert_equal(layer.weights.shape, [2, 4])
        np.testing.assert_equal(layer.biases.shape, [1, 4])
        layer.forward_propagate(X)
    
    @staticmethod
    def test_weights_in_range():
        layer = FullyConnectedLayer(2, 3)
        np.testing.assert_equal(np.max(layer.weights) < FullyConnectedLayer.SCALE, True)
        np.testing.assert_equal(np.min(layer.weights) >= -FullyConnectedLayer.SCALE, True)   
        
    @staticmethod
    def test_biases_in_range():
        layer = FullyConnectedLayer(2, 3)
        np.testing.assert_equal(np.max(layer.biases) < FullyConnectedLayer.SCALE, True)
        np.testing.assert_equal(np.min(layer.biases) >= -FullyConnectedLayer.SCALE, True) 
        