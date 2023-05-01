
import numpy as np
import matplotlib.pyplot as plt



# genere_dataset_uniform:
def genere_dataset_uniform(p,n,inf,sup):
    #np.random.seed(seed)
    lignes = np.random.uniform(inf,sup,(n,p))
    labels = np.asarray([-1 for i in range(n//2)] + [+1 for i in range(0,n//2)])
    labels = labels.reshape(-1,1)
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
    data2_label = data2_label.reshape(-1,)

    data2_negatifs = data2_desc[data2_label == neg]
    # Extraction des exemples de classe +1:
    data2_positifs = data2_desc[data2_label == pos]
    plt.scatter(data2_positifs[:,0],data2_positifs[:,1],marker='x', label="classe "+str(pos),color="blue") # 'o' rouge pour la classe -1
    plt.scatter(data2_negatifs[:,0],data2_negatifs[:,1],marker='o', label="classe "+str(neg),color="red") # 'x' bleu pour la classe +1
    
    plt.legend()
    plt.show()

  
# generate dataset XOR
def create_XOR(dim,s):
    data_gauss_desc, data_gauss_label = genere_dataset_gaussian(np.array([-1,1]),np.array([[s,0],[0,s]]),np.array([-1,-1]),np.array([[s,0],[0,s]]),dim,0,1)
    data_gauss_desc1, data_gauss_label1 = genere_dataset_gaussian(np.array([1,-1]),np.array([[s,0],[0,s]]),np.array([1,1]),np.array([[s,0],[0,s]]),dim,0,1)
    data_desc = np.vstack((data_gauss_desc,data_gauss_desc1))
    data_label = np.concatenate((data_gauss_label,data_gauss_label1),axis=0)
    data_label = data_label.reshape(-1,1)
    return data_desc,data_label

def create_deriv_Z_par_W(inpu,d,d_prime):
    deriv_Z_par_W = np.zeros((d_prime,d_prime,d))

    for i in range(d_prime):
        deriv_Z_par_W[i,i]  = inpu
            
    return deriv_Z_par_W

def sech(X):
    return 1 / np.cosh(X)

def sigmoid(X):
    return 1/(1 + np.exp(-X))


def y_to_one_hot(y,nb_classe):
    min_y = np.min(y)
    N = y.shape[0]
    y_shift= y - min_y
    y_oh = np.zeros((N, nb_classe), dtype='int')
    y_oh[np.arange(N), y_shift] = 1
    return y_oh

def softmax(yhat):
    return np.exp(yhat) / np.sum(np.exp(yhat),axis=1,keepdims=True)