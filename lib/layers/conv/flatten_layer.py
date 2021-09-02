import numpy as np
import math

class FlattenLayer():
    
    def forward_propagate(self, X):
        self.X = X
        
        return X.flatten()
    
    def gradient(self, prev_grad):
        return prev_grad.T
    
    def is_elementwise(self):
        return False

from unittest import TestCase

class FlattenLayerTests(TestCase):
        
    @staticmethod
    def test_fp():
        X = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        exp = np.array([1, 2, 3, 4, 5, 6])
        
        layer = FlattenLayer()
        result = layer.forward_propagate(X)
        
        np.testing.assert_equal(result, exp)
