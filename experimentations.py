
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
pos_cen = (-12, -12)
pos_sig = [[1, 0], [0, 1]]

neg_cen = (-2, -2)
neg_sig = [[1, 0], [0, 1]]
data,labels = genere_dataset_gaussian(pos_cen,pos_sig,neg_cen,neg_sig,100,-1,1)
labels = labels.reshape(-1,1)


# definir un reseau linaire
'''
N = data.shape[0]

facteur_norma = 0.4
plage_biais = (0,1)
neuro_i = 2
neuro_o = 1

lineaire = ModuleLineaire(neuro_i,neuro_o,plage_biais,facteur_norma,init=1)
mseloss = MSELoss()
network_layers = [lineaire]
network = Sequentiel(network_layers)

opti = Optim(network,mseloss,1e-5)

para = opti.SGD(data,labels,1000,1000)
plot_frontiere_lineaire(data,opti.getNetwork(),1000)
plot2DSet(data,labels,-1,1)
plt.show()'''


seed = 42
np.random.seed(seed)
data_xor , labels_xor = create_XOR(220,0.1)


neuro_i_1 = 2
neuro_o_1 = 4

neuro_i_2 = 4
neuro_o_2 = 1

facteur_norma = 0.4
plage_biais = (0,1)


facteur_norma = 0.8
lineaire_1 = ModuleLineaire(neuro_i_1 ,neuro_o_1 ,plage_biais,facteur_norma,init=1)
lineaire_2 = ModuleLineaire(neuro_i_2 ,neuro_o_2 ,plage_biais,facteur_norma,init=1)
TanH = ModuleTanH()
sigmoide = ModuleSigmoide()
mseloss = MSELoss()


network_layers = [lineaire_1,TanH,lineaire_2,sigmoide]
network = Sequentiel(network_layers)


opti = Optim(network,mseloss,1e-3)

para = opti.SGD(data_xor,labels_xor,50,1000)

plot_frontiere_non_lineaire(data_xor,opti.getNetwork(),1000)
plot2DSet(data_xor,labels_xor,0,1)
plt.show()





















