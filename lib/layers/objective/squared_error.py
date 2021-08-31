import numpy as np
from .objective import Objective

class SquaredError(Objective):
    
    def _gradient(self, target, estimate):
        return 2 * (estimate - target)
        
    def _objective(self, target, estimate):
        return (target - estimate) * (target - estimate)
        
from unittest import TestCase

class SquaredErrorTest(TestCase):
    
    @staticmethod
    def test_gradient_zeros():
        target = np.zeros(3)
        pred = np.zeros(3)
        exp = np.zeros(3)
        
        layer = SquaredError()
        layer.forward_propagate(pred)
        gradient = layer.gradient(target)
        
        np.testing.assert_array_equal(gradient, exp)
    
    @staticmethod
    def test_eval_identity():
        target = np.identity(4)
        pred = np.zeros(4)
        exp = np.identity(4)
        
        layer = SquaredError()
        layer.forward_propagate(pred)
        error = layer.eval(target)
        
        np.testing.assert_array_equal(error, exp)
    
    @staticmethod
    def test_eval_zeros():
        target = np.zeros(2)
        pred = np.zeros(2)
        exp = np.zeros(2)
        
        layer = SquaredError()
        layer.forward_propagate(pred)
        error = layer.eval(target)
        
        np.testing.assert_array_equal(error, exp)