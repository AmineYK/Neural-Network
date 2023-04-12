import numpy as np
import random
from utils import *

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss(Loss):

    def __init__(self):
        pass

    def forward(self, y, yhat):
        # yhat --> la sortie de la derniere couche du reseau -> de est de taille ( N x d )
        # avec d le nombre de neurones de la derniere couche de notre reseau
        # on moyenne par rapport a tous le batch --> N 
        # np.sum(np.power((yhat - y),2)) --> sera de taille (1,d)
        return np.mean(np.power((yhat - y),2),axis=1)

    def backward(self, y, yhat):
        # matrice de taille (N x d)
        return 2 * (yhat - y)



class  Module (object):
    def __init__(self):
        super().__init__()
        

    def zero_grad(self):
        d,d_prime = self._parameters.shape
        self._gradient = np.zeros((d,d_prime))

    def forward(self, X):
        # X represente l'entrée du module
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        # faut pas oublier de deriver par rapport au biais aussi 
        # le biais sera de taille ( N x d)
        # avec d le nombre de neurone du module ( couche )
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass


class  ModuleLineaire (Module):
    def __init__(self,W,biais,init=0):
        d,d_prime = W
        dim_biais,val0,val1 = biais
        assert dim_biais == d_prime , "Dimensions incompatibles pour W et biais"
        
        self.biais = np.random.uniform(val0,val1,(1,dim_biais))
        if init == 1:
            self._parameters = np.random.uniform(0,1,W)*0.1
        else: 
            self._parameters = np.zeros(W)

        self._gradient = np.zeros((d,d_prime))
        self.zero_grad()

    
    
    def forward(self, X):
        # X represente l'entrée du module --> (N x d)
        # np.dot(X,self._gradient) --> (N , d')
        # d --> le nombre de neurone du module precedent 
        # d' --> le nombre de neurone du module actuelle 
        # (N , d') + (N , d')
        return np.dot(X,self._parameters)

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient
        self.zero_grad()

    def backward_update_gradient(self, input, delta):
        # faut pas oublier de deriver par rapport au biais aussi 
        # le biais sera de taille ( N x d)
        # avec d le nombre de neurone du module ( couche )
        ## Met a jour la valeur du gradient
        # sortie ---> (N x d*d')
        # delta ----> (N x d_prime)

        # pour des questions de securité
        assert input.shape[0] == delta.shape[0] , "Taille du batch non prise en compte"
        # dans le cas ou delta --> (N,) autrement dit : d_prime = 1
        if delta.ndim == 1:
            delta = delta.reshape(-1,1)

        n , d = input.shape
        _ , d_prime = delta.shape


        for i_inp in range(input.shape[0]):
            deriv_Z_par_W = create_deriv_Z_par_W(input[i_inp],d,d_prime)
            # deriv_Z_par_W ---> (d_prime,d_prime,d)
            # delta[i_inp] ---> (d_prime,)
            # self._gradient --> (d_prime,d)
            delta[i_inp] = delta[i_inp].reshape(1,-1)
            self._gradient +=  np.dot(delta[i_inp],deriv_Z_par_W ).reshape(d_prime,d).T
    
 

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur

        # pour des questions de securité
        assert input.shape[0] == delta.shape[0] , "Taille du batch non prise en compte"
        # dans le cas ou delta --> (N,) autrement dit : d_prime = 1
        if delta.ndim == 1:
            delta = delta.reshape(-1,1)

        return np.dot(delta,self._parameters.T)
          


class  ModuleTanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # X represente l'entrée du module
        return np.tanh(X)

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass


class  ModuleSigmoide(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # X represente l'entrée du module
        return 1/(1 + np.exp(-X))

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass