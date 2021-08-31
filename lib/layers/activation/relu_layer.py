import numpy as np
from .activation_layer import ActivationLayer

class ReluLayer(ActivationLayer):
    '''
    A simple implementation of a RelU activation layer that works on both vertices and matrices.
    The RelU function is max(0, x).
    '''
    
    def forward_propagate(self, X, otypes=[np.float]):
        self.X = X
        return np.vectorize(self.g, otypes)(X)
    
    def g(self, x):
        return max(0, x)
    
    def gradient(self, otypes=[np.float]):
        def grad_z(z):
            return 1 if z >= 0 else 0
        
        return np.vectorize(grad_z, otypes)(self.X)

from unittest import TestCase

class ReluLayerTests(TestCase):
    
    @staticmethod
    def test_gradient():
        X = np.array([
            [1, 0],
            [-3, -4],
            [-99, 0.2]
        ])
        grad_exp = np.array([
            [1, 1],
            [0, 0],
            [0, 1]
        ])
        layer = ReluLayer()
        layer.forward_propagate(X)
        grad_act = layer.gradient()
        np.testing.assert_equal(grad_act, grad_exp)
        
    @staticmethod
    def test_zeros_vector():
        X = np.array([0., 0., 0., 0.])
        layer = ReluLayer()
        np.testing.assert_equal(layer.forward_propagate(X), [0., 0., 0., 0.])

    @staticmethod
    def test_zeros_matrix():
        X = np.array([
            [0., 0.],
            [0., 0.]
        ])
        layer = ReluLayer()
        np.testing.assert_equal(layer.forward_propagate(X), [ [0., 0.], [0., 0.] ])
        
    @staticmethod
    def test_matrix():
        X = np.array([
            [-1., 1.],
            [-5., 0.]
        ])
        layer = ReluLayer()
        np.testing.assert_equal(layer.forward_propagate(X), [ [0., 1.], [0., 0.] ])
                
    @staticmethod
    def test_vector():
        X = np.array([-1., -5., 60., 0.955, -0.44])
        layer = ReluLayer()
        np.testing.assert_equal(layer.forward_propagate(X), [0., 0., 60., 0.955, 0.])