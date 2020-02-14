import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from averyFunctions import CreateRandomClusters
from averyFunctions import jacobDists

X, y = CreateRandomClusters(4000)


threshhold = .3
edge_neighbors = 2

dists = jacobDists(X)
neighbors = dists < threshhold
neighbor_count = np.sum(neighbors, axis = 1)
core_indexing = neighbor_count > edge_neighbors
core_points = neighbors[core_indexing]
edge_points = neighbors[~core_indexing]

avail = core_indexing.copy()


clusters = []
while avail.any():
    ind = np.random.choice(np.where(avail)[0])
    cluster = neighbors[ind,:]
    old_cluster = np.zeros(cluster.shape)

    while np.any(cluster != old_cluster):
        old_cluster= cluster.copy()
        cluster = np.any(cluster | neighbors[cluster & core_indexing], axis = 0)

    avail = avail & ~cluster
    clusters.append(cluster)

y_hat = np.vstack(clusters)

noise = ~np.any(y_hat, axis = 0)

y_hat = np.vstack((y_hat, noise))

class DBSCAN:
    def __init__(self, threshhold, num_neighbors_edge):
        self.core_thresh = threshhold
        self.num_neighbors_edge = num_neighbors_edge

    def Fit(self, x):
        dists = jacobDists(X)
        neighbors = dists < self.core_thresh
        neighbor_count = np.sum(neighbors, axis=1)
        core_indexing = neighbor_count > self.num_neighbors_edge
        avail = core_indexing.copy()
        clusters = []
        while avail.any():
            ind = np.random.choice(np.where(avail)[0])
            cluster = neighbors[ind, :]
            old_cluster = np.zeros(cluster.shape)

            while np.any(cluster != old_cluster):
                old_cluster = cluster.copy()
                cluster = np.any(cluster | neighbors[cluster & core_indexing], axis=0)

            avail = avail & ~cluster
            clusters.append(cluster)

        y_hat = np.vstack(clusters)

        noise = ~np.any(y_hat, axis=0)

        y_hat = np.vstack((y_hat, noise))
        self.core_points = x[core_indexing]
        self.core_class = np.argmax(y_hat, axis = 0)[core_indexing]
        return y_hat

    def Predict(self, x):
        diff = x.reshape(1, -1, x.shape[1]) - self.core_points.reshape(-1, 1, x.shape[1])
        dist = np.sum(diff ** 2, axis=2)
        y_hat = self.core_class[np.argmin(dist,axis=0)]
        neighbors = dists < self.core_thresh
        y_hat[~np.any(neighbors, axis = 0)] = 0
        return y_hat
            # neighbors = dists < self.core_thresh
            # neighbors_class = neighbors * self.core_class
            # unique, counts = np.unique(neighbors_class, return_counts = True, axis = 1)

dbscan = DBSCAN(threshhold, edge_neighbors)
dbscan.Fit(X)
y_hat = dbscan.Predict(X)
colors = y_hat
fig = plt.figure()
plt.scatter(X[:,0], X[:,1], c=colors, cmap='jet')
plt.show()

class GMM:
    def __init__(self, K):
        self.k = K

    def Fit(self, X, iterations = 1000):
        mean_ind = np.random.randint(len(X), size=(K, 1))
        mean_K = X[mean_ind]
        self.covs = [np.eye(x.shape[1])] * self.k


        probs = self.Probabilities(X)

        closest = np.argmax(dists, axis=1)

    def Probabilities(self, x):
        probs = []
        for i in range(self.k):
            probs.append(mvn(x, mean = self.mean_k[k], cov = self.covs[k]))
        probs = np.vstack(self.probs)

    def Predict(self, x):
        self.y_hat = np.argmax(self.probs, axis = 0)

    def Loss(self, x):
            probs = self.Probabilities(x)
            loss = np.sum(np.log(np.sum(probs, axis=1)))




def GaussianMixtureMeans(X, K, iterations = 100):
    mean_ind = np.random.randint(len(X), size=(K,1))
    mean_k = X[mean_ind]
    mean_k = mean_k.reshape(len(mean_ind),X.shape[1]) # I legitimately don't know why this is doing this, but whatever,  c'est la vie.
    likelihoods = dict()
    idx = np.arange(0, len(X), 1)
    closest = np.random.randint(K,size=(len(X)))
    for i in range(iterations):
        dists = []
        clusters = []
        for k in range(K):
            X_k = X[idx[closest==k]]
            likelihoods[k] = {"mu": mean_k[k], "SIGMA": np.cov(X_k.T)}


        for m,l in likelihoods.items():
            dists.append(mvn.logpdf(X, l["mu"], l["SIGMA"]))

        dists = np.array(dists).T
        closest = np.argmax(dists, axis=1)
        sets = []
        num_sets = []
        for j in range(len(mean_ind)):
            set_j = X[idx[closest==j]]
            clusters.append(np.mean(set_j, axis = 0))
            num_sets.append(len(set_j))

        num_sets = np.array(num_sets)
        weights = num_sets/len(X)
        mean_k = np.array(clusters)
        y_hat = closest
        loss = -np.sum( np.sum(dists * weights, axis = 1) )
    return mean_k, y_hat, loss

gmm = GMM(4)
mean_k, y_hat = gmm(X)

fig = plt.figure()
plt.scatter(X[:,0], X[:,1], c = y_hat)

