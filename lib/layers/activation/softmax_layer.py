import numpy as np
from .activation_layer import ActivationLayer

class SoftmaxLayer(ActivationLayer):
    '''
    Am implementation of the softmax activation layer.
    The softmax function is exp(x) / sum_k(exp(x_k)) for 1..K, where K is the number of features.
    '''
    
    def __init__(self, eps=1e-8):
        self.eps = eps
    
    def forward_propagate(self, X, otypes=[np.float]):
        self.X = X
        return self.g(X)
    
    def g(self, X):
        # We protect against underflow and underflow by both shifting
        # the X matrix and using the log softmax.
        # This means we can write the softmax function as:
        #   g(x) = log(e^x/sum(e^x_j))
        #        = x - log(sum(e^x_j))
        
        minimum = -700
        z = X - np.max(X, axis=1, keepdims=True)
        z[z < minimum] = minimum
        
        sum_term = np.sum(np.exp(z), axis=1, keepdims=True)
        
        return np.exp(z) / sum_term
    
    def gradient(self):
        softmax = self.g(self.X)
        return softmax * (1 - softmax)

from unittest import TestCase

class SoftmaxLayerTests(TestCase):
    
    @staticmethod
    def test_big_nums():
        X = np.array([
            [1e99, 0],
            [-1e99, 0]
        ])
        
        layer = SoftmaxLayer()
        out = layer.forward_propagate(X)
    
    @staticmethod
    def test_small_nums():
        X = np.array([
            [1e-99, 0],
            [-1e-99, 0]
        ])
        
        layer = SoftmaxLayer()
        out = layer.forward_propagate(X)
    
    @staticmethod
    def test_divide_by_zero():
        X = np.array([
            [0., 0., 0.],
            [0., 0., 0.]
        ])
        
        layer = SoftmaxLayer()
        out = layer.forward_propagate(X)

    @staticmethod
    def test_adds_to_one():
        X = np.array([
            [1., 2., 3.],
            [2., 5., 6.]
        ])
        sum_exp = np.array([
            [1.],
            [1.]
        ])
        
        layer = SoftmaxLayer()
        out = layer.forward_propagate(X)
        sum_act = np.sum(out, axis=1, keepdims=True)
                        
        np.testing.assert_array_almost_equal(sum_act, sum_exp)
        
    @staticmethod
    def test_grad_dims():
        X = np.array([
            [1., 0.],
            [0., 1.]
        ])
        layer = SoftmaxLayer()
        layer.forward_propagate(X)
        grad = layer.gradient()
        
        dim0_exp = X.shape[0]
        dim1_exp = X.shape[1]
        dim0_act = grad.shape[0]
        dim1_act = grad.shape[1]
                        
        np.testing.assert_equal(dim0_act, dim0_exp)
        np.testing.assert_equal(dim1_act, dim1_exp)

    @staticmethod
    def test_identity_matrix():
        X = np.array([
            [1., 0.],
            [0., 1.]
        ])
        Y_exp = np.array([
            [0.73105857863001, 0.26894142137],
            [0.26894142137, 0.73105857863001]
        ])
        layer = SoftmaxLayer()
        Y = layer.forward_propagate(X)
        np.testing.assert_array_almost_equal(Y, Y_exp)