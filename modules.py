import numpy as np
import math
from utils import *



class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class CrossEntropieLoss(Loss):
    def forward(self, y, yhat):

        # z ---> (N x K)
        z = softmax(yhat)
        # y_oh : matrice binaire ---> (N x K)
        y_oh = y_to_one_hot(y)
        N = y.shape[0]

        # log_sum ---> (N x 1)
        log_sum = np.log(np.sum(np.exp(z),axis=1,keepdims=True))
        indices = np.where(y_oh == 1)[1]
        # selected_entropie ---> (N x 1)
        selected_entropie = z[np.arange(N), indices].reshape(-1,1)

        # cout_cross_entropique ---> (N x 1)
        cout_cross_entropique = -selected_entropie + log_sum
        return cout_cross_entropique

    def backward(self, y, yhat):
        z = softmax(yhat)
        y_oh = y_to_one_hot(y)
        
        # de taille (N x K)
        return z - y_oh



class MSELoss(Loss):

    def __init__(self):
        pass

    def forward(self, y, yhat):
        # yhat --> la sortie de la derniere couche du reseau -> de est de taille ( N x d )
        # avec d le nombre de neurones de la derniere couche de notre reseau
        # on moyenne par rapport a tous le batch --> N 
        # np.sum(np.power((yhat - y),2)) --> sera de taille (1,d)
        #loss  = np.mean(np.power((y - yhat),2),axis=1)
        loss = np.linalg.norm(y - yhat , axis=1)**2
        if loss.ndim == 1:
            loss = loss.reshape(-1,1)

        return loss 

    def backward(self, y, yhat):
        # matrice de taille (N x d)
        return -2 * (y - yhat)



class  Module (object):
    def __init__(self):
        super().__init__()
        

    def zero_grad(self):
        d,d_prime = self._parameters.shape
        self._gradient_parametres = np.zeros((d,d_prime))
        #self._gradient_biais = np.zeros((N,d_prime))

    def forward(self, X):
        # X represente l'entrée du module
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

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
    def __init__(self,W,facteur_norma,init=0):
        d,d_prime = W
        #dim_biais0,dim_biais1,val0,val1 = biais
        #assert dim_biais1 == d_prime , "Dimensions incompatibles pour W et biais"

        if init == 1:
            #self._parameters = np.random.uniform(0,1,W) * facteur_norma
            self._parameters = 2 * ( np.random.rand(d,d_prime) - 0.5 )
            #self.biais = np.random.uniform(val0,val1,(dim_biais0,dim_biais1))

        else: 
            self._parameters = np.zeros(W)
            #self.biais = np.zeros((dim_biais0,d_prime))



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
        self._parameters -= gradient_step*self._gradient_parametres
        #self.biais -= gradient_step*self._gradient_biais


    def backward_update_gradient(self, input, delta):

        # pour des questions de securité
        assert input.shape[0] == delta.shape[0] , "Taille du batch non prise en compte"
        # dans le cas ou delta --> (N,) autrement dit : d_prime = 1
        if delta.ndim == 1:
            delta = delta.reshape(-1,1)

        n , d = input.shape
        _ , d_prime = delta.shape


        self._gradient_parametres +=  np.dot(input.T,delta)
        #self._gradient_biais += np.ones((n,d_prime))


    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur

        # pour des questions de securité
        assert input.shape[0] == delta.shape[0] , "Taille du batch non prise en compte"
        # dans le cas ou delta --> (N,) autrement dit : d_prime = 1
        if delta.ndim == 1:
            delta = delta.reshape(-1,1)

        # deriver par rapport la ieme entrée --> biais n'intervient pas
        return np.dot(delta,self._parameters.T)


class  ModuleActivation(Module):
    def __init__(self):
        super().__init__()      

          
class  ModuleTanH(ModuleActivation):
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
        # pour des questions de securité
        assert input.shape[0] == delta.shape[0] , "Taille du batch non prise en compte"
        # dans le cas ou delta --> (N,) autrement dit : d_prime = 1
        if delta.ndim == 1:
            delta = delta.reshape(-1,1)

        #deriv_tanh = np.power(sech(input),2)
        deriv_tanh = 1 / np.cosh(input) ** 2
        # produit element par element
        #return np.multiply(delta,deriv_tanh)
        #return delta * deriv_tanh
        return ( 1 - np.tanh(input)**2 ) * delta
class  ModuleSigmoide(ModuleActivation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # X represente l'entrée du module
        return sigmoid(X)

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        # pour des questions de securité
        assert input.shape[0] == delta.shape[0] , "Taille du batch non prise en compte"
        # dans le cas ou delta --> (N,) autrement dit : d_prime = 1
        if delta.ndim == 1:
            delta = delta.reshape(-1,1)
            
        deriv_sigm = sigmoid(input) * (1 - sigmoid(input))
        # dervi_sigm ----> N x d
        # delta ---> N x d_prime
        # sortie doit etre ---> N x d_prime
        # produit element par element
        #return np.multiply(delta,deriv_sigm)
        #return delta * deriv_sigm
        return ( np.exp(-input) / ( 1 + np.exp(-input) )**2 ) * delta


class  Sequentiel(object):
    def __init__(self,modules=[],lossModule=None):
        super().__init__()
        self.modules = modules
        self.lossModule = lossModule
        self.sortie_loss = []
        self.inputs = []

    def addLossModule(self,lossModule):
        self.lossModule = lossModule

    def getLossModule(self):
        return self.lossModule

    def addModule(self,module):
        self.modules.append(module)

    def getModule(self,indice):
        return self.modules[indice]

    def getModules(self):
        return self.modules

    def get_parameters(self):
        return [(module._parameters) for module in self.modules[:-1] if not isinstance(module,ModuleActivation)]


    def predict(self,data,verbose=False):
        input = data
        self.inputs = []
        for module in self.modules[:-1]:
            self.inputs.append(input)
            sortie_module = self.forward_step(input,module,verbose)
            input = sortie_module
            
        return sortie_module


    def forward_step(self,input,module,verbose):
        output = module.forward(input)
        if verbose:
            print(f"Forward on module  : {module}")
            print(f"Input : ",input.shape)
            print(f"Output : ",output.shape)
            print()
        return output


    def forward(self,data,labels,verbose=False):
        input = data
        self.inputs = [data]
        for module in self.modules[:-1]:
            sortie_module = self.forward_step(input,module,verbose)
            self.inputs.append(sortie_module)
            input = sortie_module
            
        assert labels.shape == sortie_module.shape , f"Dernier module {self.modules[-1]} incompatible avec le module Loss"

        sortie_loss = self.modules[-1].forward(labels,self.inputs[-1])
        self.inputs.append(sortie_loss)

        return sortie_loss


    def backward_step(self,input,delta,module,eps,verbose):

        # si c'est un module d'activation : faire uniquement le retro-propagation de l'erreur
        if isinstance(module,ModuleActivation):
            delta_sortie = module.backward_delta(input,delta)
        else:
            module.backward_update_gradient(input,delta)
            module.update_parameters(eps)
            module.zero_grad()
            delta_sortie = module.backward_delta(input,delta)

        delta_sortie = module.backward_delta(input,delta)
        if verbose:
            print(f"Backward on module  : {module}")
            print(f"Delta Input : ",delta.shape)
            print(f"Delta Output: ",delta_sortie.shape)
            print()
        return delta_sortie


    def backward(self,labels,eps,verbose=False):

        sortie_loss = self.inputs[-1]
        assert sortie_loss.ndim != 0 and self.inputs != [] and len(self.inputs)-1 == len(self.modules), "Pass Forward pas encore faite ou mal faite!!!"

        modules = self.modules
        inputs  = self.inputs

        # self.inputs --> [X , yhat1,yhat2,...,yhatN,sortie_loss(erreur_moy)]

        delta = self.modules[-1].backward(labels,self.inputs[-2])
        if verbose:
            print(f"Backward on module  : {self.modules[-1]}")
            print(f"Delta Output : ",delta.shape)
            print()

        for i_input in range(len(self.inputs)-2,0,-1):
            module = modules[i_input-1]
            delta_sortie = self.backward_step(inputs[i_input-1],delta,module,eps,verbose)
            delta = delta_sortie




class  Optim(object):
    def __init__(self,network,loss,eps):
        super().__init__()
        self.network = network
        self.loss = loss
        self.eps = eps

    def getNetwork(self):
        return self.network

    def step(self,data,labels,verbose=False):
        sortie_loss_network = self.network.forward(data,labels,verbose)
        self.network.backward(labels,sortie_loss_network,self.eps,verbose)
        return sortie_loss_network


def SGD(network,data,labels,taille_batch,nb_epochs,verbose=False):

    batch_x = None
    batch_y = None

    loss = network.getLossModule()
    eps = 0.01
    opti = Optim(network,loss,eps)
    losses = []
    
    for i_epoch in range(nb_epochs):
        loss_epoch = opti.step(data,labels,verbose)
        losses.append(loss_epoch.mean())

        #if verbose:
        print(f"Epoche {i_epoch} , loss : {loss_epoch.mean()} ")
    

    return opti.getNetwork().get_parameters()
