import numpy as np
from .objective import Objective

class CrossEntropy(Objective):
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def _objective(self, target, estimate):
        return -np.sum(target * np.log(estimate + self.eps), axis=1, keepdims=True)
        
    def _gradient(self, target, estimate):
        return - target / (estimate + self.eps)
        
from unittest import TestCase

class CrossEntropyTests(TestCase):
    # Example cases taken from:
    # https://machinelearningmastery.com/cross-entropy-for-machine-learning/
    
    @staticmethod
    def test_distributions_against_each_other():
        y_act = np.array([
            [0.80, 0.15, 0.05],
            [0.10, 0.40, 0.50]
        ])
        y_pred = np.array([
            [0.10, 0.40, 0.50],
            [0.80, 0.15, 0.05]
        ])
        
        loss_exp = np.array([
            [2.014169],
            [2.279028]
        ])
        
        error = CrossEntropy()
        error.forward_propagate(y_pred)
        loss_act = error.eval(y_act)

        np.testing.assert_array_almost_equal(loss_act, loss_exp)
    
    @staticmethod
    def test_distributions_against_self():
        y_act = np.array([
            [0.10, 0.40, 0.50],
            [0.80, 0.15, 0.05]
        ])
        y_pred = np.array([
            [0.10, 0.40, 0.50],
            [0.80, 0.15, 0.05]
        ])
              
        loss_exp = np.array([
            [0.943348],
            [0.612869]
        ])
        
        error = CrossEntropy()
        error.forward_propagate(y_pred)
        loss_act = error.eval(y_act)

        np.testing.assert_array_almost_equal(loss_act, loss_exp)

    @staticmethod
    def test_identity():
        y_act = np.identity(4)
        y_pred = np.identity(4)
        
        loss_exp = np.array([
            [-1.442695e-08],
            [-1.442695e-08],
            [-1.442695e-08],
            [-1.442695e-08]
        ])
        
        error = CrossEntropy()
        error.forward_propagate(y_pred)
        loss_act = error.eval(y_act)

        np.testing.assert_array_almost_equal(loss_act, loss_exp)
    
    @staticmethod
    def test_zeros():
        y_act = np.zeros((1, 3))
        y_pred = np.zeros((1, 3))
        
        loss_exp = np.zeros((1, 1))
        
        error = CrossEntropy()
        error.forward_propagate(y_pred)
        loss_act = error.eval(y_act)
        
        np.testing.assert_array_almost_equal(loss_act, loss_exp)
        
    
    @staticmethod
    def test_gradient_zeros():
        y_act = np.zeros((3, 3))
        y_pred = np.zeros((3, 3))
        
        grad_exp = np.zeros((3, 3))
        
        error = CrossEntropy()
        error.forward_propagate(y_pred)
        grad_act = error.gradient(y_act)
        
        np.testing.assert_array_almost_equal(grad_act, grad_exp)
  