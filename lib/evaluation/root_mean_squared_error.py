import numpy as np

def root_mean_squared_error(Y_act, Y_pred):
    '''
    An implementation of the Root-Mean-Square Error function.
    See here:
    https://en.wikipedia.org/wiki/Root-mean-square_deviation
    '''
    N = Y_act.shape[0]

    return np.sqrt(
        (1/N) * (Y_act - Y_pred).T @ (Y_act - Y_pred)
    )[0][0]

from unittest import TestCase

class RMSETests(TestCase):
    
    @staticmethod
    def test_zeros():
        y_act = np.zeros((3,3))
        y_pred = np.zeros((3,3))
        
        root_mean_squared_error(y_act, y_pred)