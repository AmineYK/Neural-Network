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
        self._gradient_biais = np.zeros((1,d_prime))

    def forward(self, X):
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
    def __init__(self,d,d_prime,plage_biais,facteur_norma,init=0):

        val0,val1 = plage_biais

        if init == 1:
            #self._parameters = np.random.uniform(-1,1,(d,d_prime))
            #self.biais = np.random.uniform(val0,val1,(1,d_prime)) 
            self._parameters = 2 * (np.random.rand(d,d_prime) - 0.5)
            self.biais = np.random.randn(1,d_prime)
        else: 
            self._parameters = np.zeros((d,d_prime))
            self.biais = np.zeros((1,d_prime))

        self.zero_grad()

    
    
    def forward(self, X):
        return np.dot(X,self._parameters) + self.biais

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient_parametres
        self.biais -= gradient_step*self._gradient_biais


    def backward_update_gradient(self, input, delta):

        # pour des questions de securité
        assert input.shape[0] == delta.shape[0] , "Taille du batch non prise en compte"
        # dans le cas ou delta --> (N,) autrement dit : d_prime = 1
        if delta.ndim == 1:
            delta = delta.reshape(-1,1)

        n , d = input.shape
        _ , d_prime = delta.shape


        self._gradient_parametres +=  np.dot(input.T,delta)
        self._gradient_biais += np.sum(delta,axis=0)


    

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

    def zero_grad(self):
        pass 

    def update_parameters(self, gradient_step=0.001):
        pass 

    def backward_update_gradient(self, input, delta):
        pass


          
class  ModuleTanH(ModuleActivation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # X represente l'entrée du module
        return np.tanh(X)


    def backward_delta(self, input, delta):
        # pour des questions de securité
        assert input.shape[0] == delta.shape[0] , "Taille du batch non prise en compte"
        # dans le cas ou delta --> (N,) autrement dit : d_prime = 1
        if delta.ndim == 1:
            delta = delta.reshape(-1,1)

        return ( 1 - np.tanh(input)**2 ) * delta


class  ModuleSigmoide(ModuleActivation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # X represente l'entrée du module
        return sigmoid(X)

    def backward_delta(self, input, delta):
        # pour des questions de securité
        assert input.shape[0] == delta.shape[0] , "Taille du batch non prise en compte"
        # dans le cas ou delta --> (N,) autrement dit : d_prime = 1
        if delta.ndim == 1:
            delta = delta.reshape(-1,1)

        sig =self.forward(input)
        return sig * (1 - sig) * delta


class  Sequentiel(object):
    def __init__(self,modules=[]):
        super().__init__()
        self.modules = modules
        self.inputs = []

    def addModule(self,module):
        self.modules.append(module)

    def getModule(self,indice):
        return self.modules[indice]

    def getModules(self):
        return self.modules

    def get_parameters(self):
        return [(module._parameters) for module in self.modules if not isinstance(module,ModuleActivation)]


    def predict(self,data):
        input = data
        self.inputs = []
        for module in self.modules:
            self.inputs.append(input)
            sortie_module = self.forward_step(input,module)
            input = sortie_module
            
        return sortie_module


    def forward_step(self,input,module):
        return module.forward(input)
        
    def forward(self,data):
        input = data
        self.inputs = []
        for module in self.modules:
            self.inputs.append(input)
            sortie_module = self.forward_step(input,module)
            input = sortie_module
        
        return sortie_module


    def backward_step(self,input,delta,module):

        # si c'est un module d'activation : faire uniquement le retro-propagation de l'erreur
        if isinstance(module,ModuleActivation):
            delta_sortie = module.backward_delta(input,delta)
        else:
            module.backward_update_gradient(input,delta)
            delta_sortie = module.backward_delta(input,delta)

        return delta_sortie


    def backward(self,delta):

        assert len(self.inputs) == len(self.modules), "Pass Forward pas encore faite ou mal faite!!!"

        modules = self.modules
        inputs  = self.inputs

        # self.inputs --> [X , yhat1,yhat2,...,yhatN]
        
        for i_input in range(len(self.modules)-1,-1,-1):
            module = modules[i_input]
            delta_sortie = self.backward_step(inputs[i_input],delta,module)
            delta = delta_sortie

    def update_all(self,eps):
        for module in self.modules:
            module.update_parameters(eps)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()


class  Optim(object):
    def __init__(self,network,loss,eps):
        super().__init__()
        self.network = network
        self.loss = loss
        self.eps = eps

    def getNetwork(self):
        return self.network

    def step(self,data,labels):
        y_pred = self.network.forward(data)
        loss_value = self.loss.forward(labels,y_pred)

        delta_final = self.loss.backward(labels,y_pred)

        self.network.zero_grad()
    
        self.network.backward(delta_final)
        self.network.update_all(self.eps)

        return loss_value.mean()

    def SGD(self,data,labels,taille_batch,nb_epochs):

        n_samples = data.shape[0]
        n_batches = n_samples // taille_batch

        losses = []

        for epoch in range(nb_epochs):
            loss_epoch = 0

            indices = np.random.permutation(n_samples)
            data = data[indices]
            labels = labels[indices]

            data_batchs = np.array_split(data, n_batches)
            labels_batchs = np.array_split(labels, n_batches)

            for data_batch, labels_batch in zip(data_batchs, labels_batchs):
                loss_epoch += self.step(data_batch, labels_batch)

            loss_epoch /= n_batches
            losses.append(loss_epoch)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{taille_batch} - Loss: {loss_epoch}")

        return losses
