
import numpy as np
import matplotlib.pyplot as plt


# genere_dataset_uniform:
def genere_dataset_uniform(p,n,inf,sup):
    #np.random.seed(seed)
    lignes = np.random.uniform(inf,sup,(n,p))
    labels = np.asarray([-1 for i in range(n//2)] + [+1 for i in range(0,n//2)])
    return lignes,labels

# genere_dataset_gaussian:
def genere_dataset_gaussian(pos_cen,pos_sig,neg_cen,neg_sig,n,neg,pos):
    data_neg = np.random.multivariate_normal(neg_cen,neg_sig,n)
    data_pos = np.random.multivariate_normal(pos_cen,pos_sig,n)
    res = np.concatenate([data_neg,data_pos])
    labels = np.array([neg for i in range(n)] + [pos for i in range(0,n)])
    return res,labels


# genere_dataset_gaussian_bis:
def genere_dataset_gaussian_bis(pos_cen,pos_sig,neg_cen,neg_sig,n):
    data_neg = np.random.multivariate_normal(neg_cen,neg_sig,n)
    data_pos = np.random.multivariate_normal(pos_cen,pos_sig,n)
    res = np.concatenate([data_pos,data_neg])
    labels = np.array([1 for i in range(n)] + [-1 for i in range(0,n)])

# plot2DSet:
def plot2DSet(data2_desc,data2_label,neg,pos):
    # Extraction des exemples de classe -1:
    data2_negatifs = data2_desc[data2_label == neg]
    # Extraction des exemples de classe +1:
    data2_positifs = data2_desc[data2_label == pos]
    plt.scatter(data2_negatifs[:,0],data2_negatifs[:,1],marker='x', color="blue") # 'o' rouge pour la classe -1
    plt.scatter(data2_positifs[:,0],data2_positifs[:,1],marker='o', color="red") # 'x' bleu pour la classe +1
    plt.show()
# plot_frontiere_lineaire:
def plot_frontiere_lineaire(desc_set, classifier,step=30):
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    #res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res = classifier.forward(grid)
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])

# plot_frontiere_non_lineaire
def plot_frontiere_non_lineaire(desc_set, network,step=30):
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    #res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    lineaire_1,TanH,lineaire_2,sigmoide = network
    sortie_lineaire_1 = lineaire_1.forward(grid)
    activation_sortie_lineaire_1 = TanH.forward(sortie_lineaire_1)
    print(activation_sortie_lineaire_1)
    sortie_lineaire_2 = lineaire_2.forward(activation_sortie_lineaire_1)
    sig = sigmoide.forward(sortie_lineaire_2)
    # produite le vecteur de prediction
    res  = np.where(sig >= 0.5, 1, 0)
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])

# generate dataset XOR
def create_XOR(dim,s):
    data_gauss_desc, data_gauss_label = genere_dataset_gaussian(np.array([-1,1]),np.array([[s,0],[0,s]]),np.array([-1,-1]),np.array([[s,0],[0,s]]),dim,0,1)
    data_gauss_desc1, data_gauss_label1 = genere_dataset_gaussian(np.array([1,-1]),np.array([[s,0],[0,s]]),np.array([1,1]),np.array([[s,0],[0,s]]),dim,0,1)
    data_desc = np.vstack((data_gauss_desc,data_gauss_desc1))
    data_label = np.concatenate((data_gauss_label,data_gauss_label1),axis=0)
    return data_desc,data_label

def create_deriv_Z_par_W(inpu,d,d_prime):
    deriv_Z_par_W = np.zeros((d_prime,d_prime,d))
    # pour tous le batch
#     for n in range(N):
#         # i de 0 a 3 (pour tous les d_prime)
    for i in range(d_prime):
        deriv_Z_par_W[i,i]  = inpu
            
    return deriv_Z_par_W