import numpy as np

def accuracy(y_true, y_pred):
    '''
    This function computes the accuracy of a set of class predictions against
    the samples' true classes.
    '''
    N = y_true.shape[0]
    return (y_true == y_pred).sum() / N

from unittest import TestCase

class AccuracyTests(TestCase):
    
    @staticmethod
    def test_accuracy_1():
        y_act = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        
        np.testing.assert_equal(1., accuracy(y_act, y_pred))
    
    @staticmethod
    def test_accuracy_0():
        y_act = np.array([0, 1, 0])
        y_pred = np.array([1, 0, 1])
        
        np.testing.assert_equal(0., accuracy(y_act, y_pred))