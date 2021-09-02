import numpy as np
import math

class ConvolutionalLayer():
    
    SCALE = 1e-4
    LEARNING_RATE = 1e-4
    
    def __init__(self, shape, M, learning_rate=LEARNING_RATE, scale=SCALE):
        self.M = M
        self.H, self.W = shape
        self.learning_rate = learning_rate
       
        # create the filter
        self.K = (np.random.rand(M, M) - 0.5) * scale
    
    def out_shape(self):
        return (self.H - self.M + 1, self.W - self.M + 1)
    
    @staticmethod
    def _cross_correlate(X, K):
        
        H, W = X.shape
        M, M = K.shape
        
        out_dims = (H - M + 1, W - M + 1)
        out = np.empty(out_dims)
        
        out_a, out_b = out_dims
        
        for a in range(out_a):
            for b in range(out_b):
                x = a - math.floor(M/2) + 1
                y = b - math.floor(M/2) + 1
                
                X_sub = X[x:x+M, y:y+M]
                out[a,b] = np.sum(X_sub*K)
                
        return out
    
    def forward_propagate(self, X):
        self.X = X
        
        return self._cross_correlate(X, self.K)
    
    
    def local_gradient(self, prev_grad):
        # create gradient matrix
        grad_dims = (self.M, self.M)
        grad = np.empty(grad_dims)
        
        for a in range(self.M):
            for b in range(self.M):
                af = self.H - self.M + a + 1
                bf = self.W - self.M + b + 1
                
                X_sub = self.X[a:af,b:bf]
                grad[a,b] = np.sum(prev_grad*X_sub)
        
        return grad
    
    def gradient(self, prev_grad):
        local_grad = self.local_gradient(prev_grad)
        
        return self._cross_correlate(self.K.T, local_grad)
    
    def is_elementwise(self):
        return False
    
    def update(self, prev_grad):
        grad = self.local_gradient(prev_grad)
        
        #new_grad_dims = (self.W, self.W)
        #new_grad = np.empty(new_grad_dims)
        
        #for a in range(self.W):
        #    for b in range(self.W):
        #        new_grad[a,b] = self._cross_correlate(prev_grad, grad)
        
        self.K = self.K - self.learning_rate * grad
        
        return grad
    
    def _update_weights(self, grad):
        N = self.X.shape[0]
        grad_weights = self.X.T@grad
        self.weights = self.weights - self.learning_rate * grad_weights
        
    def _update_biases(self, grad):
        N = self.X.shape[0]
        grad_biases = np.ones([N,1]).T@grad
        self.biases = self.biases - (self.learning_rate) * grad_biases

from unittest import TestCase

class ConvolutionalLayerTests(TestCase):
        
    @staticmethod
    def test_out_shape():
        X = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        filter_size = 3
        
        layer = ConvolutionalLayer(X.shape, 3)
        
        np.testing.assert_equal(layer.K.shape, (filter_size, filter_size))
    
    @staticmethod
    def test_filter_shape():
        X = np.array([
            [1,   2,  3,  4],
            [5,   6,  7,  8],
            [9,  10, 11, 12],
            [13, 14, 15, 16],
        ])
        filter_size = 3
        
        layer = ConvolutionalLayer(X.shape, 3)
        
        np.testing.assert_equal(layer.K.shape, (3, 3))
     
    @staticmethod
    def test_fp():
        X = np.array([
            [1, 2, 3, 4],
            [2, 2, 3, 2],
            [1, 3, 3, 3],
            [4, 4, 4, 4],
        ])
        K = np.array([
            [1, 2, 3],
            [2, 2, 3],
            [1, 3, 3],
        ])
        filter_size = 3
        
        exp = np.array([
            [50, 57],
            [60, 63]
        ])
        
        layer = ConvolutionalLayer(X.shape, 3)
        layer.K = K
        
        np.testing.assert_equal(layer.forward_propagate(X), exp)
      
    @staticmethod
    def test_fp_example():
        X = np.array([i for i in range(1, 65)]).reshape((8, 8))
        K = np.array([i for i in range(1, 10)]).reshape((3, 3))
        filter_size = 3
        
        exp = np.array([
            [ 600.,  645.,  690.,  735.,  780.,  825.],
            [ 960., 1005., 1050., 1095., 1140., 1185.],
            [1320., 1365., 1410., 1455., 1500., 1545.],
            [1680., 1725., 1770., 1815., 1860., 1905.],
            [2040., 2085., 2130., 2175., 2220., 2265.],
            [2400., 2445., 2490., 2535., 2580., 2625.],
        ])
        
        layer = ConvolutionalLayer(X.shape, 3)
        layer.K = K
        
        np.testing.assert_equal(layer.forward_propagate(X), exp)
               
                  
    @staticmethod
    def test_bp_example():
        # TODO: Is this actually broken?
        X = np.array([i for i in range(1, 65)]).reshape((8, 8))
        K = np.array([i for i in range(1, 10)]).reshape((3, 3))
        
        filter_size = 3
        
        delta = np.zeros((6,6))
        delta[2,2] = 1
        delta[5,2] = 2
        delta[2,5] = 3
        delta[5,5] = 4
        
        exp = np.array([
            [355., 365., 375.],
            [435., 445., 455.],
            [515., 525., 535.],
        ])

        layer = ConvolutionalLayer(X.shape, 3)
        layer.K = K
        
        layer.forward_propagate(X)
        grad = layer.update(delta)
        
        np.testing.assert_equal(grad, exp)