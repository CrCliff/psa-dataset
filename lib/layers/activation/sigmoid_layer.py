import numpy as np
from .activation_layer import ActivationLayer

class SigmoidLayer(ActivationLayer):
    '''
    A simple implementation of a sigmoid activation layer that works on both vectors and matrices.
    The sigmoid function is 1 / (1 + exp(-x)).
    '''
    
    def forward_propagate(self, X, otypes=[np.float]):
        self.X = X
        return self.g(X)
    
    def g(self, x):
        return 1. / (1. + np.exp(-x))
    
    def gradient(self):
        def grad_z(z):
            return np.multiply(self.g(z), np.add(1, np.multiply(-1, self.g(z))))
                    
        return np.apply_along_axis(grad_z, 1, self.X)

from unittest import TestCase

class SigmoidLayerTests(TestCase):
        
    @staticmethod
    def test_grad():
        X = np.array([
            [1., 0.],
            [0., 1.]
        ])
        grad_exp = np.array([
            [0.1966119332414818525374, 0.25],
            [0.25, 0.1966119332414818525374]
        ])
        layer = SigmoidLayer()
        layer.forward_propagate(X)
        grad_act = layer.gradient()
                
        np.testing.assert_array_almost_equal(grad_act, grad_exp)
    
    @staticmethod
    def test_grad_dims():
        X = np.array([
            [1., 0.],
            [0., 1.]
        ])
        layer = SigmoidLayer()
        layer.forward_propagate(X)
        grad = layer.gradient()
        
        dim0_exp = X.shape[0]
        dim1_exp = X.shape[1]
        dim0_act = grad.shape[0]
        dim1_act = grad.shape[1]
                
        np.testing.assert_equal(dim0_act, dim0_exp)
        np.testing.assert_equal(dim1_act, dim1_exp)
        
    @staticmethod
    def test_zeros_vector():
        X = np.array([0., 0., 0., 0.])
        layer = SigmoidLayer()
        np.testing.assert_equal(layer.forward_propagate(X), [0.5, 0.5, 0.5, 0.5])
        
    @staticmethod
    def test_zeros_matrix():
        X = np.array([
            [0., 0.],
            [0., 0.]
        ])
        layer = SigmoidLayer()
        np.testing.assert_equal(layer.forward_propagate(X), [ [0.5, 0.5], [0.5, 0.5] ])
        
    @staticmethod
    def test_large_vector():
        X = np.array([-1e2, 1e2])
        layer = SigmoidLayer()
        np.testing.assert_array_almost_equal(layer.forward_propagate(X), [0., 1.])
        
    