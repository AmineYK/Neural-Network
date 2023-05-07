
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pydot





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
def genere_dataset_gaussian_bis(pos_cen,pos_sig,neg_cen,neg_sig,n,neg,pos):
    data_neg = np.random.multivariate_normal(neg_cen,neg_sig,n)
    data_pos = np.random.multivariate_normal(pos_cen,pos_sig,n)
    res = np.concatenate([data_pos,data_neg])
    labels = np.array([pos for i in range(0,2*n)])
    return res,labels

# plot2DSet:
def plot2DSet(data2_desc,data2_label,neg,pos):
    # Extraction des exemples de classe -1:
    data2_label = data2_label.reshape(-1,)

    data2_negatifs = data2_desc[data2_label == neg]
    # Extraction des exemples de classe +1:
    data2_positifs = data2_desc[data2_label == pos]
    plt.title("Nuage de points")
    plt.scatter(data2_positifs[:,0],data2_positifs[:,1],marker='x', label="classe "+str(pos),color="blue") # 'o' rouge pour la classe -1
    plt.scatter(data2_negatifs[:,0],data2_negatifs[:,1],marker='o', label="classe "+str(neg),color="red") # 'x' bleu pour la classe +1
    plt.legend()

  
# generate dataset XOR
def create_XOR(dim,s):
    data_gauss_desc, data_gauss_label = genere_dataset_gaussian(np.array([-1,1]),np.array([[s,0],[0,s]]),np.array([-1,-1]),np.array([[s,0],[0,s]]),dim,0,1)
    data_gauss_desc1, data_gauss_label1 = genere_dataset_gaussian(np.array([1,-1]),np.array([[s,0],[0,s]]),np.array([1,1]),np.array([[s,0],[0,s]]),dim,0,1)
    data_desc = np.vstack((data_gauss_desc,data_gauss_desc1))
    data_label = np.concatenate((data_gauss_label,data_gauss_label1),axis=0)
    data_label = data_label.reshape(-1,1)
    return data_desc,data_label

# generate dataset XOR
def create_data_dirac(dim,s):
    data_gauss_desc, data_gauss_label = genere_dataset_gaussian_bis(np.array([-1,1]),np.array([[s,0],[0,s]]),np.array([-1,-1]),np.array([[s,0],[0,s]]),dim,0,1)
    data_gauss_desc1, data_gauss_label1 = genere_dataset_gaussian_bis(np.array([1,-1]),np.array([[s,0],[0,s]]),np.array([1,1]),np.array([[s,0],[0,s]]),dim,0,1)
    data_gauss_desc2 =  np.random.multivariate_normal(np.array([0,0]),np.array([[s,0],[0,s]]),2*dim)
    data_gauss_label2 = np.array([0 for _ in range(2*dim)])
    data_desc2 = np.vstack((data_gauss_desc,data_gauss_desc1))
    data_desc = np.vstack((data_desc2,data_gauss_desc2))
    data_label2 = np.concatenate((data_gauss_label,data_gauss_label1),axis=0)
    data_label = np.concatenate((data_label2,data_gauss_label2),axis=0)
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

def affiche_image(X_train,num,title):
    plt.title(title)
    plt.imshow(X_train[num].reshape(16,16))
    plt.show


def similarity(data,data_bis,seuil=0.5):
    return np.where(np.abs(data - data_bis).mean(axis=1) < seuil, 1, 0).mean()


def normalisation(data):
    dt = data.copy()
    for i in range(data.shape[1]):
        mini = np.min(data[:,i])
        maxi = np.max(data[:,i])
        if maxi == mini:
            if maxi > 0:
                dt[:,i] = 1
            else:
                dt[:,i] = 0
        else:dt[:,i] = (data[:,i] - mini) / (maxi - mini)
    return dt


def get_images(folder_dir):
    folder_dir = "data/pepper"
    images = []
    for path in os.listdir(folder_dir):
        # recuperer uniquement un seul canal(b) de couleur 
        ima = np.array(Image.open(folder_dir+"/"+path))[:,:,2]
        # from (256,256)  to 65536
        images.append(ima.flatten())
    return np.array(images)



def generate_noise(shape,law='normal', mean=0, std=1):

    if law == 'poisson':
        lam = mean
        return np.random.poisson(lam,shape)
    return np.random.normal(mean, std, shape)
    
def cluster_purity(labels_true, labels_pred,nb_classes=None):

    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    assert labels_true.shape == labels_pred.shape  
    
    # Trouver les étiquettes les plus communes dans chaque cluster
    clusters = np.unique(labels_pred)
    n = len(labels_true)
    if nb_classes == None : nb_classes = len(np.unique(labels_true))
    counts = np.zeros((len(clusters),nb_classes ))
    for i, c in enumerate(clusters):
        mask = labels_pred == c
        labels = labels_true[mask]
        counts[i, :] = np.bincount(labels, minlength=len(counts[i, :]))
    
    # Trouver la pureté en utilisant les étiquettes les plus communes
    purity = np.sum(np.max(counts, axis=1)) / n
    
    return purity



def cara_to_int(labels):
    chars = np.unique(labels)

    l = []

    for let in labels:
        for i,cara in enumerate(chars):
            if let == cara:
                l.append(i)

    return np.array(l)



def net_to_graph(net, net_name="network", horizontal=False):

    modules = net.modules
    n = len(modules)
    modules_list = np.array_split(modules, np.ceil(n / 6))

    if os.path.exists(net_name):    
        os.rmdir(net_name)

    os.mkdir(net_name)
    
    base = 0
    for i, modules in enumerate(modules_list):
        net_to_graph_step(modules, net_name, i, horizontal, base)
        base += len(modules)




def net_to_graph_step(liste_modules,name,nb,horizontal, base):


    if horizontal:
        graph = pydot.Dot(graph_type="digraph", rankdir="LR")
    else:
        graph = pydot.Dot(graph_type="digraph")

    for i, layer in enumerate(liste_modules):
        label = f"{base + i} - {layer}"

        if layer.__class__.__name__ in ["ModuleLineaire", "Conv1D", "MaxPool1D", "Flatten"]:
            node = pydot.Node(label, shape="box")
        else:
            node = pydot.Node(label)

        graph.add_node(node)

    nodes = graph.get_nodes()

    for i in range(len(nodes) - 1):
        src_node = nodes[i]
        dst_node = nodes[i + 1]
        edge = pydot.Edge(src_node, dst_node)
        graph.add_edge(edge)

    graph.write_png(f"{name}/{nb}.png")



