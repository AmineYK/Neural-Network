
from utils import *
from modules import *
import numpy as np 
from numpy import linalg as LA



#######################################################
#################### LINEAIRE #########################
#######################################################



# boucle d'apprentissage 
def apprentissage(data,labels,network,seuil_norme_param,nb_epoche,verbose=False):

    lineaire,mseloss = network
    input = data
    epoche = 1
    liste_parametres = []
    liste_gradients = [LA.norm(lineaire._gradient)]

    while epoche < nb_epoche:

        param = lineaire._parameters

        # ajout du nouveau parametre
        # ATTENTION : POINTEUR
        liste_parametres.append(param.copy())
        
        
        # calcule de la sortie du module lineaire
        sortie_lin = lineaire.forward(input)
        #print(sortie_lin.shape)
        # calcule de la sortie du module MSE
        sortie_loss = mseloss.forward(labels,sortie_lin)
        #print(sortie_loss)

        # definir l'erreur de correction en retro-propagation
        delta_fin = mseloss.backward(labels,sortie_loss)

        # retro-propager l'erreur sur le module lineaire 
        lineaire.backward_update_gradient(data,delta_fin)
        # _gradient vient d'etre mis à jour
        liste_gradients.append(LA.norm(lineaire._gradient))

        # mettre à jour les parametres W
        lineaire.update_parameters()
        # _parametres vient d'etre mis à jour


        # verifier le critere de convergence
        if LA.norm(param - lineaire._parameters) < seuil_norme_param and epoche != 1:
           break

        if verbose:
            print(f"Epoche  {epoche} :  {param}")

        # passer à l'epoche suivante
        epoche += 1


        # non utilisable vuque c'est la premiere couche
        '''delta_lin = lineaire.backward_delta(data,delta_fin)
        input = delta_lin'''
        #print(input)

    return liste_parametres,liste_gradients




# definir un jeu de données lineairement separable 
seed = 42
np.random.seed(seed)
pos_cen = (2, 2)
pos_sig = [[1, 0], [0, 1]]

neg_cen = (-2, -2)
neg_sig = [[1, 0], [0, 1]]
data,labels = genere_dataset_gaussian(pos_cen,pos_sig,neg_cen,neg_sig,100,-1,1)

#plot2DSet(data,labels,-1,1)

# definir une couche lineaire
biais  = []
W = (2,1) 
lineaire =  ModuleLineaire(W,biais,init=1)

# definir une couche de sortie de type MSE 
mseloss = MSELoss()


# Test 
# reseau : une liste de module dans l'ordre d'apparition 
network = [lineaire,mseloss]
parametres , gradients= apprentissage(data,labels,network,0.001,10,verbose=False)
'''print("Les parametres")
print(parametres)
param_opt = parametres[-1]
print("\n W optimal \n",)

print("\n Les normes des gradients")
print(gradients)

lineaire_optimal =  ModuleLineaire(W,biais,init=0)
lineaire_optimal._parameters = param_opt
plot_frontiere_lineaire(data,lineaire_optimal,step=30)
plot2DSet(data,labels,-1,1)'''




#### SYNTHESE 
'''
Un problème courant lié à des normes de gradients trop grandes est le phénomène du "gradient explosif", 
où les gradients deviennent très grands pendant l'entraînement, provoquant une instabilité du modèle et une mauvaise performance. 
Cela se produit souvent dans les réseaux de neurones profonds avec des fonctions d'activation non linéaires, 
car les gradients peuvent être amplifiés à chaque couche lors de la rétropropagation du gradient.

Il existe plusieurs techniques pour éviter que les normes des gradients ne deviennent trop grandes :

Normalisation des entrées : Normaliser les données d'entrée du modèle peut aider à éviter que les gradients ne deviennent trop grands. 
Cela peut être fait en divisant les données par leur écart-type ou en les mettant à l'échelle dans une plage spécifique.

Utilisation de fonctions d'activation adaptées : Certaines fonctions d'activation, comme ReLU (Rectified Linear Unit) 
et ses variantes, sont moins sensibles au problème du gradient explosif par rapport à d'autres fonctions
d'activation, comme la fonction sigmoïde ou la tangente hyperbolique.

Utilisation de techniques de régularisation : Des techniques de régularisation, telles que la régularisation L1 ou L2, 
peuvent aider à limiter la taille des poids et ainsi réduire la magnitude des gradients.

Utilisation d'optimiseurs adaptatifs : Les optimiseurs adaptatifs, tels que l'optimiseur Adam ou RMSprop,
peuvent ajuster automatiquement les taux d'apprentissage en fonction des normes des gradients, ce qui peut aider à éviter 
des mises à jour de poids trop importantes.
'''





###########################################################
#################### NON LINEAIRE #########################
###########################################################


def apprentissage_non_lineaire(data_xor,labels_xor,network,seuil_norme_param,nb_epoche,verbose=False):

    lineaire_1,TanH,lineaire_2,sigmoide,mseloss = network
    input = data
    epoche = 1

    while epoche < nb_epoche:

        # pass FORWARD
        sortie_lineaire_1 = lineaire_1.forward(data_xor)
        activation_sortie_lineaire_1 = TanH.forward(sortie_lineaire_1)
        sortie_lineaire_2 = lineaire_2.forward(activation_sortie_lineaire_1)
        activation_sortie_lineaire_2 = sigmoide.forward(sortie_lineaire_2)
        sortie_loss = mseloss.forward(labels_xor,activation_sortie_lineaire_2)

        # pass BACKWARD
        
        # MODULE MSE
        delta_fin = mseloss.backward(labels_xor,sortie_loss)
        
        # MODULE LINEAIRE 2
        lineaire_2.backward_update_gradient(activation_sortie_lineaire_1,delta_fin)
        lineaire_2.update_parameters()
        delta_2 = lineaire_2.backward_delta(activation_sortie_lineaire_1,delta_fin)

        # MODULE LIEAIRE 1
        lineaire_1.backward_update_gradient(data_xor,delta_2)
        lineaire_1.update_parameters()
        # pas necessaire 
        delta_1 = lineaire_1.backward_delta(data_xor,delta_2)

        if verbose:
            print(f"Epoche  {epoche} : \n param_couche1 : \n {lineaire_1._parameters} \n param_couche2 : \n {lineaire_2._parameters}")
            print()

        # passer à l'epoche suivante
        epoche += 1



    return lineaire_1._parameters , lineaire_2._parameters




data_xor , labels_xor = create_XOR(50,0.01)
#plot2DSet(data_xor,labels_xor,0,1)

W_1=(2,2)
W_2=(2,1)
biais = None
lineaire_1 = ModuleLineaire(W_1,biais,init=1)
lineaire_2 = ModuleLineaire(W_2,biais,init=1)
TanH = ModuleTanH()
sigmoide = ModuleSigmoide()
mseloss = MSELoss()

network = [lineaire_1,TanH,lineaire_2,sigmoide,mseloss]
parametres_1,parametres_2 = apprentissage_non_lineaire(data_xor,labels_xor,network,0.001,100,verbose=True)

print(parametres_1)
print(parametres_2)
lineaire_1_optimal =  ModuleLineaire(W_1,biais,init=1)
lineaire_1_optimal._parameters = parametres_1

lineaire_2_optimal =  ModuleLineaire(W_2,biais,init=1)
lineaire_2_optimal._parameters = parametres_2

network_optimal = [lineaire_1_optimal,TanH,lineaire_2_optimal,sigmoide]
plot_frontiere_non_lineaire(data_xor,network_optimal,step=30)
plot2DSet(data_xor,labels_xor,0,1)