"""
Python implementation to reproduce the experiments in the paper

Minimum information trees for high dimensional data visualization in clustering   
Author: Alexandre L. M. Levada

ABSTRACT

Visualizing the qualitative outcomes of clustering algorithms in high-dimensional spaces remains a persistent challenge
in data analysis and machine learning. Traditional dimensionality reduction techniques often distort the underlying 
structure of clusters or fail to provide interpretable representations of inter-cluster relationships. In this paper, 
we introduce Minimum Information Trees (MINFO trees), an information-theoretic, graph-based method for visualizing 
high-dimensional data with an emphasis on preserving clustering structure. By leveraging pairwise information measures 
and constructing information-theoretic based k-NN graphs, MINFO trees generate data visualizations that reflect both 
local cohesion and global separation among clusters. Our method provides interpretable and faithful representations of
clustering results, enabling qualitative evaluation of cluster quality and relationships. Experimental results on 
real-world datasets highlight the differences between MINFO trees over traditional dimensionality reduction methods 
such as t-SNE and UMAP in terms of preserving cluster topology and enhancing visual interpretability.

"""
import warnings
import umap
import matplotlib as mpl
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.neighbors as sknn
import sklearn.datasets as skdata
import sklearn.utils.graph as sksp
from sklearn import preprocessing
from sklearn import metrics
from numpy import inf
from scipy import optimize
from scipy.signal import medfilt
from networkx.convert_matrix import from_numpy_array
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import v_measure_score

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Build the KNN graph
def build_KNN_Graph(dados, k):
    # Build a graph
    CompleteGraph = sknn.kneighbors_graph(dados, n_neighbors=n-1, mode='distance')
    # Adjacency matrix
    W_K = CompleteGraph.toarray()
    # NetworkX format
    K_n = nx.from_numpy_array(W_K)
    # MST
    W_mst = nx.minimum_spanning_tree(K_n)
    mst = [(u, v, d) for (u, v, d) in W_mst.edges(data=True)]
    mst_edges = []
    for edge in mst:
        edge_tuple = (edge[0], edge[1], edge[2]['weight'])
        mst_edges.append(edge_tuple)
    # Create the k-NNG
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=nn, mode='distance')
    # Adjacency matrix
    W = knnGraph.toarray()
    # NetworkX format
    G = nx.from_numpy_array(W)
    # To assure the k-NNG is connected we add te MST edges
    G.add_weighted_edges_from(mst_edges)
    # Convert to adjacency matrix
    A = nx.to_numpy_array(G)  
    return A

# Build the KNN graph
def build_Complete_Graph(dados):
    # Generate complete graph
    n = dados.shape[0]
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=n-1, mode='distance')
    A = knnGraph.toarray()  
    return A

# Optional function to normalize the curvatures to the interval [0, 1]
def normalize_curvatures(curv):
    if curv.max() != curv.min():
        k = 0.001 + (curv - curv.min())/(curv.max() - curv.min())
    else:
        k = curv
    return k

# Plot the KNN graph
def plot_KNN_graph(A, target, layout, K=0):
	# Create a networkX graph object
	G = from_numpy_array(A)
	color_map = []
	for i in range(A.shape[0]):
	    if type(K) == list:
	        if K[i] > 0:
	    	    color_map.append('black')
	        else:
	            if target[i] == -1:
	            	color_map.append('black')
	            elif target[i] == 0:
	            	color_map.append('blue')
	            elif target[i] == 1:
	            	color_map.append('red')
	            elif target[i] == 2:
	            	color_map.append('green')
	            elif target[i] == 3:
	            	color_map.append('purple')
	            elif target[i] == 4:
	            	color_map.append('orange')
	            elif target[i] == 5:
	            	color_map.append('magenta')
	            elif target[i] == 6:
	            	color_map.append('darkkhaki')
	            elif target[i] == 7:
	            	color_map.append('brown')
	            elif target[i] == 8:
	            	color_map.append('salmon')
	            elif target[i] == 9:
	            	color_map.append('cyan')
	            elif target[i] == 10:
	            	color_map.append('darkcyan')
	    else:
	        if target[i] == -1:
	        	color_map.append('black')
	        elif target[i] == 0:
	        	color_map.append('blue')
	        elif target[i] == 1:
	        	color_map.append('red')
	        elif target[i] == 2:
	        	color_map.append('green')
	        elif target[i] == 3:
	        	color_map.append('purple')
	        elif target[i] == 4:
	        	color_map.append('orange')
	        elif target[i] == 5:
	        	color_map.append('magenta')
	        elif target[i] == 6:
	        	color_map.append('darkkhaki')
	        elif target[i] == 7:
	        	color_map.append('brown')
	        elif target[i] == 8:
	        	color_map.append('salmon')
	        elif target[i] == 9:
	        	color_map.append('cyan')
	        elif target[i] == 10:
	        	color_map.append('darkcyan')
	plt.figure(1)
	# Há vários layouts, mas spring é um dos mais bonitos
	if layout == 'spring':
	    pos = nx.spring_layout(G, iterations=50)
	else:
	    pos = nx.kamada_kawai_layout(G) # ideal para plotar a árvore!
	nx.draw_networkx(G, pos, node_size=25, node_color=color_map, with_labels=False, width=0.25, alpha=0.4)
	plt.show()

# Compute the free energy
def free_energy():
	n = A.shape[0]
	free_energy = 0
	for i in range(n):
		neighbors = A[i, :]
		indices = neighbors.nonzero()[0]
		labels = target[indices]
		uim = np.count_nonzero(labels==target[i])
		free_energy += uim
	return free_energy

# Defines the pseudo-likelihood function
def pseudo_likelihood(beta):
	n = A.shape[0]
	# Computes the free energy
	free = free_energy()
	# Computes the number of labels (states of the Potts model)
	c = len(np.unique(target))
	# Computes the expected energy
	expected = 0
	for i in range(n):
		neighbors = A[i, :]
		indices = neighbors.nonzero()[0]
		labels = target[indices]
		num = 0
		den = 0
		for k in range(c):
			u = np.count_nonzero(labels==k)
			e = np.exp(beta*u)
			num += u*e
			den += e
		expected += num/den
	# Calculates the PL function value
	PL = free - expected
	return PL

# Compute the first and second order Fisher local information
def FisherInformation(A, beta):
	n = A.shape[0]
	# Computes the number of labels (states of the Potts model)
	c = len(np.unique(target))
	PHIs = np.zeros(n)
	PSIs = np.zeros(n)
	for i in range(n):
		neighbors = A[i, :]
		indices = neighbors.nonzero()[0]
		labels = target[indices]
		uim = np.count_nonzero(labels==target[i])
		Uis = np.zeros(c)
		vi =  np.zeros(c)
		wi = np.zeros(c)
		Ai = np.zeros((c, c))
		Bi = np.zeros((c, c))
		# Build vectors vi and wi
		for k in range(c):
			Uis[k] = np.count_nonzero(labels==k)
			vi[k] = uim - Uis[k]
			wi[k] = np.exp(beta*Uis[k])
		# Build matrix A
		for k in range(c):
			Ai[:, k] = Uis
		# Build matrix B
		for k in range(c):
			for l in range(c):
				Bi[k, l] = Uis[k] - Uis[l]  
		# Compute the first and second order Fisher information
		PHIs[i] = np.sum( np.kron((vi*wi), (vi*wi).T) ) / np.sum( np.kron(wi, wi.T) )
		Li = Ai*Bi
		Mi = np.reshape(np.kron(wi, wi.T), (c, c))
		PSIs[i] = np.sum( Li*Mi ) / np.sum( np.kron(wi, wi.T) )
	return (PHIs, PSIs)

# Compute the information graph
def InformationGraph(A, K, alpha):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i,j] > 0:
                if target[i] == target[j]:
                    A[i,j] = alpha*(K[i] + K[j])
                else:
                    A[i,j] = K[i] + K[j]
    return A

# Generate the minimum information tree
def MinimumInformationTree(H):
    # Networkx graph
    G = nx.from_numpy_array(H)
    # MST
    T = nx.minimum_spanning_tree(G)
    # Adjacency matrix
    R = nx.to_numpy_array(T)
    return R

# Computes the evaluation metrics
def compute_metrics(A, labels):
    rotulos = np.unique(labels)    
    comunidades = []
    for c in rotulos:        
        community = np.where(labels==c)[0]
        comunidades.append(community)
    #G = nx.from_numpy_array(A)
    modula = nx.community.modularity(A, comunidades)
    coverage, performance = nx.community.partition_quality(A, comunidades)
    cc = nx.average_clustering(A)
    return (modula, coverage, performance, cc)

##############################################
############# Beginning of the script
##############################################
#X = skdata.load_iris()
#X = skdata.fetch_openml(name='led24', version=1)
#X = skdata.load_digits()
X = skdata.fetch_openml(name='vowel', version=1)
#X = skdata.fetch_openml(name='texture', version=1)
#X = skdata.fetch_openml(name='mfeat-fourier', version=1)
#X = skdata.fetch_openml(name='mfeat-pixel', version=1)
#X = skdata.fetch_openml(name='mfeat-factors', version=1)
#X = skdata.fetch_openml(name='mfeat-karhunen', version=1)
#X = skdata.fetch_openml(name='mfeat-morphological', version=1)
#X = skdata.fetch_openml(name='satimage', version=1)
#X = skdata.fetch_openml(name='USPS', version=1) 
#X = skdata.fetch_openml(name='JapaneseVowels', version=1)
#X = skdata.fetch_openml(name='prnn_crabs', version=1) 
#X = skdata.fetch_openml(name='Smartphone-Based_Recognition_of_Human_Activities', version=1)
#X = skdata.fetch_openml(name='micro-mass', version=1) # Muito bom!
#X = skdata.fetch_openml(name='ecoli', version=1)                  
#X = skdata.fetch_openml(name='collins', version=4)                
#X = skdata.fetch_openml(name='energy-efficiency', version=1) 
#X = skdata.fetch_openml(name='artificial-characters', version=1)
#X = skdata.fetch_openml(name='nursery', version=1)
#X = skdata.fetch_openml(name='one-hundred-plants-shape', version=1) 
#X = skdata.fetch_openml(name='one-hundred-plants-texture', version=1)
#X = skdata.fetch_openml(name='leukemia', version=1)
#X = skdata.fetch_openml(name='UMIST_Faces_Cropped', version=1)
#X = skdata.fetch_openml(name='letter', version=1)
#X = skdata.fetch_openml(name='spectrometer', version=1)
#X = skdata.fetch_openml(name='leaf', version=1)
#X = skdata.fetch_openml(name='abalone', version=1)
#X = skdata.fetch_openml(name='primary-tumor', version=1)
#X = skdata.fetch_openml(name='soybean', version=1)
#X = skdata.fetch_openml(name='cardiotocography', version=1)

dados = X['data']
target = X['target']

# Convert labels to integers
S = list(set(target))
target = np.array(target)
for i in range(len(target)):
	for k in range(len(S)):
		if target[i] == S[k]:
			target[i] = k
		else:
			continue

# Convert to integers
target = target.astype('int32')

# Reduce large datasets
if dados.shape[0] >= 50000:
    dados, _, target, _ = train_test_split(dados, target, train_size=0.025, random_state=42)
elif dados.shape[0] >= 10000:
    dados, _, target, _ = train_test_split(dados, target, train_size=0.1, random_state=42)
elif dados.shape[0] >= 5000:
    dados, _, target, _ = train_test_split(dados, target, train_size=0.2, random_state=42)
elif dados.shape[0] > 2500:
    dados, _, target, _ = train_test_split(dados, target, train_size=0.5, random_state=42)

ground_truth = target.copy()

# For opemML datasets - categorical data 
if not isinstance(dados, np.ndarray):
    cat_cols = dados.select_dtypes(['category']).columns
    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
    # Convert to numpy (openml uses dataframe)
    dados = dados.to_numpy()
    #target = target.to_numpy()

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

print('N = ', n)
print('M = ', m)
print('C = %d' %c)

# Fixed number of neighbors
nn = 15		# fixo

print('K = ', nn)
print()

# Remove nan's
dados = np.nan_to_num(dados)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

# Clustering algorithms
clustering = AgglomerativeClustering(n_clusters=c, linkage='ward').fit(dados)
target = clustering.labels_
target = target.astype('int32')

# Clustering evaluation metrics
rand = rand_score(ground_truth, target)
fm = fowlkes_mallows_score(ground_truth, target)
vm = v_measure_score(ground_truth, target)
print('Rand index: ', rand)
print('Fowlkes-Mallows score: ', fm)
print('V-measure score: ', vm)
print()

# Build the adjacency matrix of the graph
A = build_KNN_Graph(dados, nn)
G = nx.from_numpy_array(A)
# Regular MST
T = nx.minimum_spanning_tree(G)
# Adjacency matrix
R = nx.to_numpy_array(T)
# Plot minimum spanning tree
plot_KNN_graph(R, target, layout='kawai')

# Estimates the maximum pseudo-likelihood estimator of the inverse temperature
sol = optimize.root_scalar(pseudo_likelihood, method='secant', x0=0, x1=1)
if not sol.converged:
    sol = optimize.root_scalar(pseudo_likelihood, method='brentq', bracket=[-3, 3])
print('MPL estimator: ', sol)
print()
# Maximum pseudo-likelihood estimator
beta_mpl = sol.root
# Compute the first and second order local Fisher information 
PHI, PSI = FisherInformation(A, beta_mpl)
# Approximate the local curvatures
curvaturas = -PSI/(PHI+0.001)
# Normalize curvatures
K = normalize_curvatures(curvaturas)

# Information graph
alpha = (np.sqrt(5) - 1)/2
IG = InformationGraph(A, K, alpha=alpha)
# Minimum information tree
MinT = MinimumInformationTree(IG)
# Plot minimum information tree
plot_KNN_graph(MinT, target, layout='kawai')

# Compute metrics in the regular MST
MST_modula, MST_coverage, MST_performance, MST_cc = compute_metrics(T, target)
centrality = nx.information_centrality(T)
print('Modularity (MST): ', MST_modula)
print('Coverage (MST): ', MST_coverage)
print('Performance (MST): ', MST_performance)
print('Centrality (MST): ', sum(list(centrality.values())))
print()

# Compute metrics in the regular MST
MIT = nx.from_numpy_array(MinT)
MIT_modula, MIT_coverage, MIT_performance, MIT_cc = compute_metrics(MIT, target)
centrality = nx.information_centrality(MIT)
print('Modularity (MIT): ', MIT_modula)
print('Coverage (MIT): ', MIT_coverage)
print('Performance (MIT): ', MIT_performance)
print('Centrality (MIT): ', sum(list(centrality.values())))
print()

# Performance of clustering after UMAP
model = TSNE(n_components=2, random_state=42)
tsne_data = model.fit_transform(dados)
# Clustering in low dimensional data
clustering = AgglomerativeClustering(n_clusters=c, linkage='ward').fit(tsne_data)
target = clustering.labels_
target = target.astype('int32')

# Build the adjacency matrix of the graph
A = build_Complete_Graph(tsne_data)
G = nx.from_numpy_array(A)
# Regular MST
T = nx.minimum_spanning_tree(G)
# Adjacency matrix
R = nx.to_numpy_array(T)

# Compute metrics in the regular UMAP data MST
TMST_modula, TMST_coverage, TMST_performance, TMST_cc = compute_metrics(T, target)
centrality = nx.information_centrality(T)
print('Modularity (t-SNE MST): ', TMST_modula)
print('Coverage (t-SNE MST): ', TMST_coverage)
print('Performance (t-SNE MST): ', TMST_performance)
print('Centrality (t-SNE MST): ', sum(list(centrality.values())))
print()

# Performance of clustering after UMAP
model = umap.UMAP(n_components=2, random_state=42)
umap_data = model.fit_transform(dados)
# Clustering in low dimensional data
clustering = AgglomerativeClustering(n_clusters=c, linkage='ward').fit(umap_data)
target = clustering.labels_
target = target.astype('int32')

# Build the adjacency matrix of the graph
A = build_Complete_Graph(umap_data)
G = nx.from_numpy_array(A)
# Regular MST
T = nx.minimum_spanning_tree(G)
# Adjacency matrix
R = nx.to_numpy_array(T)
# Plot minimum spanning tree
plot_KNN_graph(R, target, layout='kawai')

# Compute metrics in the regular UMAP data MST
UMST_modula, UMST_coverage, UMST_performance, UMST_cc = compute_metrics(T, target)
centrality = nx.information_centrality(T)
print('Modularity (UMAP MST): ', UMST_modula)
print('Coverage (UMAP MST): ', UMST_coverage)
print('Performance (UMAP MST): ', UMST_performance)
print('Centrality (UMAP MST): ', sum(list(centrality.values())))
print()
