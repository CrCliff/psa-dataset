import numpy as np
from .objective import Objective

class LogLoss(Objective):
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def _objective(self, target, estimate):
        return -(target * np.log(estimate + self.eps) + (1 - target) * np.log(1 - estimate + self.eps))
        
    def _gradient(self, target, estimate):
        return ((1 - target) / (1 - estimate + self.eps)) - (target / (estimate + self.eps))

from unittest import TestCase

class LogLossTest(TestCase):
    
    @staticmethod
    def test_eval_dims():
        y_act = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        y_pred = np.array([
            [0.4, 0.5, 0.6],
            [0.1, 0.2, 0.3]
        ])
        
        error = LogLoss()
        error.forward_propagate(y_pred)
        out = error.eval(y_act)
        
        np.testing.assert_equal(out.shape, (2, 3))
        
    @staticmethod
    def test_zeros():
        y_act = np.zeros((3,3))
        y_pred = np.zeros((3,3))
        
        error = LogLoss()
        error.forward_propagate(y_pred)
        error.eval(y_act)
    
    @staticmethod
    def test_gradient_zeros():
        y_act = np.zeros((3,3))
        y_pred = np.zeros((3,3))
        exp = np.ones((3,3))
        
        layer = LogLoss()
        layer.forward_propagate(y_pred)
        gradient = layer.gradient(y_act)
        
        np.testing.assert_array_almost_equal(gradient, exp)
    
    @staticmethod
    def test_eval_zeros():
        y_act = np.zeros((2,2))
        y_pred = np.zeros((2,2))
        exp = np.zeros((2,2))
        
        layer = LogLoss()
        layer.forward_propagate(y_pred)
        error = layer.eval(y_act)
        
        np.testing.assert_array_almost_equal(error, exp)