import numpy as np
from .objective import Objective

class Logistic(Objective):
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def _objective(self, target, estimate):
        return -np.log(estimate)
        
    def _gradient(self, target, estimate):
        return -1 / (estimate + self.eps)

from unittest import TestCase

class LogisticTest(TestCase):
    
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
        
        error = Logistic()
        error.forward_propagate(y_pred)
        out = error.eval(y_act)
        
        np.testing.assert_equal(out.shape, (2, 3))
        
    @staticmethod
    def test_zeros():
        y_act = np.zeros((3,3))
        y_pred = np.zeros((3,3))
        
        error = Logistic()
        error.forward_propagate(y_pred)
        error.eval(y_act)
