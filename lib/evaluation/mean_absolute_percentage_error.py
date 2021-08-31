import numpy as np

def mean_absolute_percentage_error(Y_act, Y_pred, eps=1e-8):
    '''
    An implementation of the Mean Absolute Percentage Error function.
    See here:
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    '''
    N = Y_act.shape[0]

    return (1 / N) * np.sum(np.absolute(
        (Y_act - Y_pred) / (Y_act + eps)
    ))

from unittest import TestCase

class MAPETests(TestCase):
    
    @staticmethod
    def mean_absolute_percentage_error():
        y_act = np.zeros((3,3))
        y_pred = np.zeros((3,3))
        
        mean_absolute_percentage_error(y_act, y_pred)