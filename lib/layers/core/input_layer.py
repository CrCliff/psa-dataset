import numpy as np

class InputLayer:
    '''
    An implementation of the input layer for a network.
    The layer is initialized with means and standard deviations for each feature.
    The result of forward propagation is the z-scored data.
    '''
    
    def __init__(self, X, zscore=True):
        self.zscore = zscore
        if zscore:
            self.meanX = np.mean(X, axis=0)
            self.stdX = np.std(X, axis=0)
            self.stdX[np.abs(self.stdX)<=1e-8] = 1
    
    def forward_propagate(self, X):
        return ((X - self.meanX) / self.stdX
            if self.zscore else X)

from unittest import TestCase

class InputLayerTests(TestCase):
        
    @staticmethod
    def test_identity_is_mean_correct():
        X = np.array([
            [1, 0],
            [0, 1]
        ])
        layer = InputLayer(X)
        np.testing.assert_equal(layer.meanX, [0.5, 0.5])
        
    @staticmethod
    def test_identity_is_std_correct():
        X = np.array([
            [1, 0],
            [0, 1]
        ])
        layer = InputLayer(X)
        np.testing.assert_equal(layer.stdX, [0.5, 0.5])
        
    @staticmethod
    def test_is_mean_correct():
        X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        layer = InputLayer(X)
        np.testing.assert_equal(layer.meanX, [4., 5., 6.])
        
    @staticmethod
    def test_is_std_correct():
        X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        layer = InputLayer(X)
        np.testing.assert_array_almost_equal(layer.stdX, [2.44948974, 2.44948974, 2.44948974])
    
    @staticmethod
    def test_identity_forwardprop():
        X = np.array([
            [1, 0],
            [0, 1]
        ])
        X_std = np.array([
            [1, -1],
            [-1, 1]
        ])
        layer = InputLayer(X)
        np.testing.assert_equal(layer.forward_propagate(X), X_std)
    
    @staticmethod
    def test_forwardprop():
        X = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        X_std = np.array([
            [-1.22474487, -1.22474487],
            [ 0.        ,  0.        ],
            [ 1.22474487,  1.22474487]
        ])
        layer = InputLayer(X)
        np.testing.assert_array_almost_equal(layer.forward_propagate(X), X_std)