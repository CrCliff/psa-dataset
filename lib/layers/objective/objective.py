from abc import ABC, abstractmethod

class Objective(ABC):
    
    def forward_propagate(self, pred):
        self.pred = pred
        return pred
    
    def eval(self, target):
        return self._objective(target, self.pred)
    
    def gradient(self, target):
        return self._gradient(target, self.pred)
    
    @abstractmethod
    def _gradient(self, target, estimate):
        pass
    
    @abstractmethod
    def _objective(self, target, estimate):
        pass