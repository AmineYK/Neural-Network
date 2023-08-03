import numpy as np
from utils import *
from tqdm import tqdm
import copy



class  Module (object):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass



###############################################################
########################## MODULES LOSS########################
###############################################################

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass
    def __str__(self) -> str:
        pass




class CrossEntropieLoss(Loss):

    def __init__(self,nb_classe) -> None:
        super().__init__()
        self.nb_classe = nb_classe

    def forward(self, y, yhat):


        y_oh = y_to_one_hot(y,self.nb_classe)
        N = y.shape[0]

        log_sum = np.log(np.sum(np.exp(yhat),axis=1,keepdims=True))
        indices = np.where(y_oh == 1)[1]

        selected_entropie = yhat[np.arange(N), indices].reshape(-1,1)

        cout_cross_entropique = -selected_entropie + log_sum
        return cout_cross_entropique

    def backward(self, y, yhat):
        y_oh = y_to_one_hot(y,self.nb_classe)
        
        return yhat - y_oh

    def __str__(self):
        return "Cross Entropie"


class BinaryCrossEntropie(Loss):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        eps = 1e-7
        yhat =np.clip(yhat,eps,1-eps)
        loss = - ( y * np.maximum(-100,np.log(yhat)) + (1-y) * np.maximum(-100,np.log(1 - yhat)) )
        if np.isnan(loss).any():
            raise Exception("NaN values exsitantes -  Binary Cross Entropie")
        
        return loss

    def backward(self, y, yhat):
        eps = 1e-3
        return - ( y /  yhat+eps) + ( (1 - y) / (1 - yhat+eps) )


    def __str__(self):
        return "Binary Cross Entropie"




class MSELoss(Loss):

    def __init__(self):
        pass

    def forward(self, y, yhat):
        loss = np.linalg.norm(y - yhat , axis=1)**2
        if loss.ndim == 1:
            loss = loss.reshape(-1,1)

        if np.isnan(loss).any():
            raise Exception("NaN values exsitantes -  Mean Square Error - MSE")

        return loss 

    def backward(self, y, yhat):
        return -2 * (y - yhat)

    def __str__(self):
        return "Mean Square Error"


####################################################################
########################## MODULE LINEAIRE ########################
####################################################################



class  ModuleLineaire(Module): 

    def __init__(self,d,d_prime,plage_biais,facteur_norma,init="uniform"):

        val0,val1 = plage_biais

        if init == "uniform":
            self._parameters = np.random.uniform(-1,1,(d,d_prime)) * facteur_norma
            self.biais = np.random.uniform(val0,val1,(1,d_prime)) 
        elif init == "zero": 
            self._parameters = np.zeros((d,d_prime))
            self.biais = np.zeros((1,d_prime))
        elif init == 'xavier':
            xavier_range = np.sqrt(6 / (d + d_prime))
            self._parameters  = np.random.uniform(-xavier_range, xavier_range, size=(d, d_prime))
            self.biais = np.random.uniform(-xavier_range, xavier_range, size=(1, d_prime))



    
    def zero_grad(self):
        d,d_prime = self._parameters.shape
        self._gradient_parametres = np.zeros((d,d_prime))
        self._gradient_biais = np.zeros((1,d_prime))
    
    
    def forward(self, X):
        if np.isinf(self._parameters).any() or np.isinf(self.biais).any() :
            raise Exception ("Inf values on parameters , learning rate too large")
        return np.dot(X,self._parameters) + self.biais

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient_parametres
        self.biais -= gradient_step*self._gradient_biais
        if np.isnan(self._parameters).any() or np.isnan(self.biais).any() :
            raise Exception("NaN values exsitantes -  Parameters Module Lineaire")
        



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


    def __str__(self):
        return "ModuleLineaire"+str(self._parameters.shape)



class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride, init="uniform"):
        super().__init__()

        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride

        self.__init_parameters__(init, (k_size, chan_in, chan_out), (chan_out))
        self.zero_grad()

    def __str__(self):
        return (
            f"Conv1D({self._k_size}, {self._chan_in}, {self._chan_out}, {self._stride})"
        )

    def zero_grad(self):
        self._gradient_parametres= np.zeros_like(self._parameters)
        self._gradient_biais= np.zeros_like(self.biais)

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step*self._gradient_parametres
        self.biais -= gradient_step*self._gradient_biais
        if np.isnan(self._parameters).any() or np.isnan(self.biais).any() :
            raise Exception("NaN values exsitantes -  Parameters Module Lineaire")

    def forward(self, X):
        assert X.shape[2] == self._chan_in, ValueError(
            "Les dimensions de X doivent être (batch, lenght, chan_in)"
        )

        batch_size, length, _ = X.shape
        dout = (length - self._k_size) // self._stride + 1
        output = np.zeros((batch_size, dout, self._chan_out))

        for i in range(dout):
            window = X[:, i * self._stride : i * self._stride + self._k_size, :]
            output[:, i, :] = np.tensordot(
                window, self._parameters["W"], axes=([1, 2], [0, 1])
            )

        if self.bias:
            output += self._parameters["b"]

        return output

    def backward_update_gradient(self, X, delta):
        _, length, chan_in = X.shape

        assert chan_in == self._chan_in, ValueError(
            "Les dimensions de X doivent être (batch, length, chan_in)"
        )

        dout = (length - self._k_size) // self._stride + 1

        assert delta.shape == (X.shape[0], dout, self._chan_out), ValueError(
            "Delta doit être de dimension (batch, (length-k_size)//stride +1, chan_out)"
        )

        for i in range(dout):
            window = X[:, i * self._stride : i * self._stride + self._k_size, :]
            self._gradient_parametres += np.tensordot(
                delta[:, i, :], window, axes=([0], [0])
            ).transpose((1, 2, 0))

        self._gradient_biais += np.sum(delta, axis=(0, 1))

    def backward_delta(self, X, delta):
        batch_size, length, chan_in = X.shape

        assert chan_in == self._chan_in, ValueError(
            "Les dimensions de X doivent être (batch, lenght, chan_in)"
        )

        dout = (length - self._k_size) // self._stride + 1

        assert delta.shape == (batch_size, dout, self._chan_out), ValueError(
            "Delta doit être de dimension (batch, (length-k_size)/stride +1, chan_out)"
        )

        delta_prev = np.zeros_like(X)

        for i in range(dout):
            delta_i = delta[:, i, :].reshape(batch_size, 1, 1, self._chan_out)

            kernel = self._parameters[::-1, :, :].reshape(
                1, self._k_size, chan_in, self._chan_out
            )

            delta_prev[
                :, i * self._stride : i * self._stride + self._k_size, :
            ] += np.sum(delta_i * kernel, axis=-1)

        return delta_prev


class MaxPool1D(ModuleLineaire):
    def __init__(self, k_size, stride):
        super().__init__()

        self._k_size = k_size
        self._stride = stride

    def __str__(self):
        return f"MaxPool1D({self._k_size}, {self._stride})"

    def zero_grad(self):
        pass  # No gradient

    def backward_update_gradient(self, X, delta):
        pass  # No gradient to update

    def update_parameters(self, gradient_step=1e-3):
        pass  # No parameters to update

    def forward(self, X):
        batch_size, length, chan_in = X.shape
        dout = (length - self._k_size) // self._stride + 1

        X_view = np.zeros((batch_size, dout, chan_in, self._k_size))

        for i in range(dout):
            X_view[:, i, :, :] = X[
                :, i * self._stride : i * self._stride + self._k_size, :
            ].transpose((0, 2, 1))

        output = np.max(X_view, axis=-1)
        return output

    def backward_delta(self, X, delta):
        batch_size, length, chan_in = X.shape
        dout = (length - self._k_size) // self._stride + 1

        assert delta.shape == (batch_size, dout, chan_in), ValueError(
            "Delta doit être de dimension (batch, (length-k_size)/stride +1, chan_in)"
        )

        out = np.zeros_like(X)

        for i in range(dout):
            start = i * self._stride
            end = start + self._k_size
            out[:, start:end, :] += delta[:, i : i + 1, :] * (
                X[:, start:end, :] == np.max(X[:, start:end, :], axis=1, keepdims=True)
            )

        return out


    
class Flatten(Module):
    def __init__(self):
        super().__init__(False)

    def zero_grad(self):
        pass  # No gradient

    def backward_update_gradient(self, X, delta):
        pass  # No gradient to update

    def update_parameters(self, gradient_step=1e-3):
        pass  # No parameters to update

    def forward(self, X):
        return X.reshape(X.shape[0], -1)

    def backward_delta(self, X, delta):
        return delta.reshape(X.shape)


#######################################################################
########################## MODULES ACTIVATIONS ########################
#######################################################################

class  ModuleActivation(Module):
    def __init__(self):
        super().__init__()    

    def zero_grad(self):
        pass 

    def update_parameters(self, gradient_step=0.001):
        pass 

    def backward_update_gradient(self, input, delta):
        pass



class SoftMax(ModuleActivation):
    def __init__(self):
        super().__init__()  

    def forward(self, X):
        # X represente l'entrée du module
        return softmax(X)

    def backward_delta(self, input, delta):
        # pour des questions de securité
        assert input.shape[0] == delta.shape[0] , "Taille du batch non prise en compte"
        # dans le cas ou delta --> (N,) autrement dit : d_prime = 1
        if delta.ndim == 1:
            delta = delta.reshape(-1,1)

        s = softmax(input)
        return s * (1 - s) * delta

    def __str__(self):
        return "SoftMax"
          
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


    def __str__(self):
        return "TanH"


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

    def __str__(self):
        return "Sigmoid"



class ReLU(Module):
    def __init__(self):
        super().__init__()

    def zero_grad(self):
        pass  # No gradient to zero

    def update_parameters(self, gradient_step=1e-3):
        pass  # No parametrs to update

    def backward_update_gradient(self, X, delta):
        pass  # No gradient to update

    def forward(self, X):
        return np.maximum(0, X)

    def backward_delta(self, X, delta):
        return np.where(X > 0, delta, 0)


##################################################################
########################## AUTRES MODULES ########################
##################################################################


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


    def predict(self,data,neg_classe=-1):
        input = data
        self.inputs = []
        for module in self.modules:
            self.inputs.append(input)
            sortie_module = self.forward_step(input,module)
            input = sortie_module

            
        if neg_classe == -1 : 
            yhat = np.where(sortie_module > 0 , 1 , -1)
        elif neg_classe == 0:
            yhat = np.where(sortie_module > 0.5 , 1 ,0)
        else: 
            nb_classes = neg_classe
            yhat = np.argmax(sortie_module,axis=1)
        
        return yhat

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

    def update_all(self,eps,regularisation=False):
        
        params = []
        nb_module_enco = len(self.modules) // 2
        if not regularisation:
            for module in self.modules:
                module.update_parameters(eps)

        else:
            for i_module in range(nb_module_enco):
                self.modules[i_module].update_parameters(eps)
                if not isinstance(self.modules[i_module] ,ModuleActivation):
                    params.append(self.modules[i_module]._parameters)

            i = nb_module_enco
            for i_module in range(len(params)):
                if not isinstance(self.modules[i] ,ModuleActivation):
                    i += 1
                else:
                    self.modules[i]._parameters = copy.deepcopy(params[i_module]).T
                    i += 1

            print(self.modules[nb_module_enco-1]._parameters.shape)
            print()
            print(self.modules[nb_module_enco+1]._parameters.shape)



    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def accuracy(self,data,labels):
        nb_classes = len(np.unique(labels))
        if nb_classes == 2:
            neg_classe = min(np.unique(labels,return_counts=True)[0])
        else:
            neg_classe = nb_classes
        yhat = self.predict(data,neg_classe)
        return (labels == yhat ).mean()


class  Optim(object):
    def __init__(self,network,loss,eps):
        super().__init__()
        self.network = network
        self.loss = loss
        self.eps = eps
        self.losses = []

    def getNetwork(self):
        return self.network

    def step(self,data,labels,regularisation=False):
        y_pred = self.network.forward(data)
        ###### POURQUOI LES LABELS !!!!
        if isinstance(self.loss ,BinaryCrossEntropie):
            loss_value = self.loss.forward(data,y_pred)
            delta_final = self.loss.backward(data,y_pred)
        else:
            loss_value = self.loss.forward(labels,y_pred)
            delta_final = self.loss.backward(labels,y_pred)

        self.network.zero_grad()
        self.network.backward(delta_final)
        self.network.update_all(self.eps,regularisation)

        return loss_value.mean()

    def SGD(self,data,labels,taille_batch,nb_epochs,verbose=False,regularisation=False):

        n_samples = data.shape[0]
        n_batches = n_samples // taille_batch


        for epoch in tqdm(range(nb_epochs)):
            loss_epoch = 0

            indices = np.random.permutation(n_samples)
            data = data[indices]
            labels = labels[indices] #********************************************

            data_batchs = np.array_split(data, n_batches)
            labels_batchs = np.array_split(labels, n_batches)#**********************************************

            for data_batch, labels_batch in zip(data_batchs, labels_batchs):
                loss_epoch += self.step(data_batch, labels_batch,regularisation)#***********************************

            loss_epoch /= n_batches
            self.losses.append(loss_epoch)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{taille_batch} - Loss: {loss_epoch}")

    def affichage(self,data,labels,step=1000):

        if len(np.unique(labels)) == 2:



            mmax=data.max(0)
            mmin=data.min(0)
            x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
            grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
            
            # calcul de la prediction pour chaque point de la grille
            neg_classe = min(np.unique(labels,return_counts=True)[0])
            yhat = self.network.predict(grid,neg_classe)
            yhat=yhat.reshape(x1grid.shape)
            # tracer des frontieres

            plt.figure(figsize=(15,5))

            plt.subplot(121)
            plt.contourf(x1grid,x2grid,yhat,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
            plot2DSet(data,labels,neg_classe,1)

            plt.subplot(122)
            plt.title("Erreur "+str(self.loss)+" moyenne")
            plt.xlabel("Nombre d'epochs")
            plt.ylabel("Erreur moyenne")
            plt.plot(np.arange(len(self.losses)),self.losses)
            plt.show()

        else:
            plt.title("Erreur "+str(self.loss)+" moyenne")
            plt.xlabel("Nombre d'epochs")
            plt.ylabel("Erreur moyenne")
            plt.plot(np.arange(len(self.losses)),self.losses)
            plt.show()

        #if not isinstance(self.loss ,BinaryCrossEntropie):
            #print("Accuracy  : ",self.network.accuracy(data,labels))


class  AutoEncodeur(object):

    def __init__(self,network,loss,regularisation=False):
        super().__init__()
        self.network = network 
        self.loss = loss
        self.opti = None
        self.layers_decodeur = None
        self.layers_encodeur = None
        self.regularisation = regularisation

    def optimisation(self,data,labels,batch_size,epochs,eps=1e-3,verbose=False):

        if verbose:
            print("Optimisation")
            print("Batch size : ",batch_size)
            print("Epochs : ",epochs)
            print("Learning rate : ",eps)


        opti = Optim(self.network,self.loss,eps)
        opti.SGD(data,labels,batch_size,epochs,False,self.regularisation)
        if verbose :opti.affichage(data,labels)

        self.opti = opti

        nb_layers = len(self.network.getModules()) // 2
        self.layers_encodeur = self.opti.getNetwork().getModules()[:nb_layers]
        self.layers_decodeur = self.opti.getNetwork().getModules()[nb_layers:]


    def encode(self,data):
        encodeur = Sequentiel(self.layers_encodeur)
        return encodeur.forward(data)


    def decode(self,data):
        decodeur = Sequentiel(self.layers_decodeur)
        return decodeur.forward(data)


