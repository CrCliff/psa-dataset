import numpy as np
from .model import Model
from overrides import overrides

# TODO: Do we need this base class?
class GAN:
    
    def __init__(self, *, gen_layers, disc_layers, gen_ratio=1, verbose=False):
        self.gen_ratio = gen_ratio
        self.gen_layers = gen_layers
        self.disc_layers = disc_layers
        
        self.G = Model(layers=gen_layers)
        self.D = Model(layers=disc_layers)
        
    def train(self, X, batch_size=0):
        if batch_size <= 0:
            # Don't batch
            y_pred, y_true = self.forward_propagate(X)
            self.back_propagate(y_true)
        else:
            # Use batch size
            N = X.shape[0]
            for i in range(0,N,batch_size):
                start = i
                end = min(i+batch_size,N)
                
                X_batch = X[start:end:]
            
                Xf, Xr, y_pred, y_true = self.forward_propagate(X_batch)
                
                self.back_propagate(Xf, y_pred, y_true)
            return Xf
    
    def random_samples(self, X):
        N_gen = X.shape[0] * self.gen_ratio

        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        
        Z_shape = (N_gen, X.shape[1])
        
        # Generate random values for input to generator
        return np.random.normal(loc=X_mean, scale=X_std, size=Z_shape)      
        
    def forward_propagate(self, Xr):
        Z = self.random_samples(Xr)
        
        Xf = self.G.forward_propagate(Z)
        
        Xf_max = np.max(Xf, axis=1).reshape((Xr.shape[0], 1))
        Xf = Xf * (255 / Xf_max)
        
        X = np.append(Xr, Xf, axis=0)
        
        # Forward propagate samples and generated
        # samples through the discriminator
        y_pred = self.D.forward_propagate(X)
        
        y_true = np.ones((Z.shape[0], 1))
        y_true = np.append(y_true, np.zeros((Z.shape[0], 1)), axis=0)
        
        return (Xf, Xr, y_pred, y_true)
    
    def back_propagate(self, Xf, y_pred, y_true):
        self.D.back_propagate(y_true)
        
        yf = y_pred[y_pred.shape[0]//2:]
        
        self.D.forward_propagate(Xf)
        
        grad = - (1 / (yf + 1e-15))
        
        for layer in self.D.layers[-2:0:-1]:
            if layer.is_elementwise():
                grad = grad * layer.gradient()
            else:
                # Don't update layers in discriminator
                grad = grad @ layer.gradient()
                
        for layer in self.G.layers[-2:0:-1]:
            if layer.is_elementwise():
                grad = grad * layer.gradient()
            else:
                if (hasattr(layer, 'weights')):
                    # Update weights using prev gradient
                    layer.update(grad)
                grad = grad @ layer.gradient()
        
        return grad
    
    def eval(self, X):
        return self.forward_propagate(X)
    
    def loss(self, X):
        Z = self.random_samples(X)
        
        Xf = self.G.forward_propagate(Z)
        
        # Forward propagate samples and generated
        # samples through the discriminator
        y_pred = self.D.forward_propagate(Xf)
        
        return -np.log(y_pred)