
from utils import *
from modules import *
import numpy as np 
from numpy import linalg as LA



#######################################################
#################### LINEAIRE #########################
#######################################################



# definir un jeu de données lineairement separable 
seed = 42
np.random.seed(seed)
pos_cen = (2, 2)
pos_sig = [[1, 0], [0, 1]]

neg_cen = (-2, -2)
neg_sig = [[1, 0], [0, 1]]
data,labels = genere_dataset_gaussian(pos_cen,pos_sig,neg_cen,neg_sig,100,-1,1)
labels = labels.reshape(-1,1)
'''data = np.array([x for x in np.linspace(-2,2,100)]).reshape(-1,1)
labels = np.array([2 * x + np.random.uniform(-1,1)  for x in np.linspace(-2,2,100)]).reshape(-1,1)
#plot2DSet(data,labels,-1,1)
plt.scatter(data,labels)
plt.show()'''


# definir un reseau linaire
'''W = (2,1)
N = data.shape[0]
biais = (N,1,1,4)
facteur_norma = 0.4
lineaire = ModuleLineaire(W,facteur_norma,init=1)
mseloss = MSELoss()
network_layers = [lineaire]
network = Sequentiel(network_layers,mseloss)

opti = Optim(network,mseloss,0.01)

para = SGD(network,data,labels,100,10,True)
print(para)
lin_opt = ModuleLineaire(W,facteur_norma,init=0)
lin_opt._parameters = para
plot_frontiere_lineaire(data,lin_opt,30)
plot2DSet(data,labels,-1,1)'''

data_xor , labels_xor = create_XOR(50,0.05)

W_1=(2,10)
W_2=(10,1)

facteur_norma = 0.8
lineaire_1 = ModuleLineaire(W_1,facteur_norma,init=1)
lineaire_2 = ModuleLineaire(W_2,facteur_norma,init=1)
TanH = ModuleTanH()
sigmoide = ModuleSigmoide()
mseloss = MSELoss()


network_layers = [lineaire_1,TanH,lineaire_2,sigmoide]
network = Sequentiel(network_layers,mseloss)

opti = Optim(network,mseloss,0.01)
opti.step(data_xor,labels_xor,verbose=False)
 
 
parametres_optimaux = SGD(network,data_xor,labels_xor,0,1000,verbose=True)
print(parametres_optimaux)


lineaire_1_opt = ModuleLineaire(W_1,facteur_norma,init=1)
lineaire_1_opt._parameters = parametres_optimaux[0]

lineaire_2_opt = ModuleLineaire(W_2,facteur_norma,init=1)
lineaire_2_opt._parameters = parametres_optimaux[1]

network_layers_opt = [lineaire_1_opt,TanH,lineaire_2_opt,sigmoide]
network_optimal = Sequentiel(network_layers_opt,mseloss)




mmax=data_xor.max(0)
mmin=data_xor.min(0)
x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],30),np.linspace(mmin[1],mmax[1],30))
grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))

# calcul de la prediction pour chaque point de la grille
#res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])

predict = network_optimal.predict(grid)
res  = np.where(predict >= 0.5, 1, 0)

# produite le vecteur de prediction

res=res.reshape(x1grid.shape)
# tracer des frontieres
# colors[0] est la couleur des -1 et colors[1] est la couleur des +1
plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
plot2DSet(data_xor,labels_xor,0,1)
























