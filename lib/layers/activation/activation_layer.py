from abc import ABC, abstractmethod

class ActivationLayer(ABC):
    '''
    The base class for an activation layer.
    Note all activation layers use element-wise multiplication.
    '''
        
    def is_elementwise(self):
        return True
