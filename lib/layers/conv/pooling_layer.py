import numpy as np
import math

class PoolingLayer():
    
    LEARNING_RATE = 1e-4
    
    def __init__(self, shape, Q, *, stride):
        self.D, self.E = shape
        self.Q = Q
        self.stride = stride
    
    def out_shape(self):
        return (
            int((self.D - self.Q) / self.stride + 1),
            int((self.E - self.Q) / self.stride + 1)
        )   
    
    def forward_propagate(self, X):
        self.X = X
                
        map_dims = (self.D, self.E)
        self.map = np.zeros(map_dims)
        
        out_dims = (
            int((self.D - self.Q) / self.stride + 1),
            int((self.E - self.Q) / self.stride + 1)
        )
        out = np.empty(out_dims)
        
        for a in range(out_dims[0]):
            for b in range(out_dims[1]):
                x0 = a * self.stride
                y0 = b * self.stride
                
                xf = x0 + self.Q
                yf = y0 + self.Q
                
                X_sub = X[x0:xf,y0:yf]
                idx = np.where(X_sub == X_sub.max())
                
                self.map[x0 + idx[0], y0 + idx[1]] = 1
                
                out[a,b] = np.max(X_sub)
        
        return out
    
    def gradient(self, prev_grad):
        mp = self.map
        idx = np.where(mp == 1)
        
        new_grad = np.zeros(mp.shape)
        
        for i, j, g in zip(idx[0], idx[1], prev_grad):
            new_grad[i,j] = g
            
        return new_grad
    
    def is_elementwise(self):
        return False

from unittest import TestCase

class PoolingLayerTests(TestCase):
        
    @staticmethod
    def test_fp():
        X = np.array([
            [1, 1, 2, 4],
            [5, 6, 7, 8],
            [3, 2, 1, 0],
            [1, 2, 3, 4],
        ])
        filter_size = 3
        
        exp = np.array([
            [6, 8],
            [3, 4]
        ])
        
        layer = PoolingLayer(X.shape, 2, stride=2)
        
        result = layer.forward_propagate(X)
        
        np.testing.assert_equal(result, exp)
         
    @staticmethod
    def test_bp():
        X = np.array([i for i in range(1, 65)]).reshape((8, 8))
                
        X = np.array([
            [ 600.,  645.,  690.,  735.,  780.,  825.],
            [ 960., 1005., 1050., 1095., 1140., 1185.],
            [1320., 1365., 1410., 1455., 1500., 1545.],
            [1680., 1725., 1770., 1815., 1860., 1905.],
            [2040., 2085., 2130., 2175., 2220., 2265.],
            [2400., 2445., 2490., 2535., 2580., 2625.],
        ])
 
        delta = np.array([1, 2, 3, 4])
        
        exp = np.zeros((6,6))
        exp[2,2] = 1
        exp[2,5] = 2
        exp[5,2] = 3
        exp[5,5] = 4
       
        layer = PoolingLayer(X.shape, 3, stride=3)
        
        layer.forward_propagate(X)
        grad = layer.gradient(delta)
        
        np.testing.assert_equal(grad, exp)
               