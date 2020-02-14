import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
#from CHI import Silhouette
def euc_norm(x, y):
    return np.sqrt(np.sum(((x - y) ** 2), axis =1))

def sigma_dbi(x):
   return (1/len(x) * (np.sum((x - np.mean(x, axis = 0)) ** 2))) ** .5


D = 2
K = 2
N = int(K * 1e3)

X0 = np.random.randn((N//K), D) + np.array([8,10])
X1 = np.random.randn((N//K),D) + np.array([0,-4])
X2 = np.random.randn((N//K),D) + np.array([-6, 2])
X3 = np.random.randn((N//K),D) + np.array([12, 2])

X = np.vstack((X0, X1, X2, X3))
y = np.array([0]*(N//K) + [1] * (N//K) + [2] * (N//K) + [3] * (N//K))
colors = y
y0 = y==0
y1 = y==1
y2 = y==2
y3 = y==3
y = np.column_stack((y0, y1, y2, y3))

mean_ind = np.random.randint(len(X), size=(K+1,1))

mean_K = X[mean_ind]

iterations = 1000

class Clustering():
    def __init__(self, K):
        self.k = K

    def Fit(self, X, iterations = 100):
        mean_ind = np.random.randint(len(X), size=(K, 1))
        self.mean_K = X[mean_ind]
        for i in range(iterations):
            dists = []
            idx = []
            clusters = []
            for i in range(len(mean_ind)):
                dists.append(euc_norm(X, self.mean_K[i]))

            dists = np.array(dists).T
            idx = np.arange(0, len(X), 1)
            closest = np.argmin(dists, axis=1)

            for i in range(len(mean_ind)):
                clusters.append(np.mean(X[idx[closest == i]], axis=0))

            self.mean_K = np.array(clusters)

        return self.mean_K

    def Predict(self, x):
        dists = []
        for i in range(len(mean_ind)):
            dists.append(euc_norm(X, self.mean_K[i]))

        dists = np.array(dists).T
        self.y_hat = np.argmin(dists, axis = 1)
        return self.y_hat

def K_means(X, K, iterations = 10):
    mean_ind = np.random.randint(len(X), size=(K, 1))
    mean_K = X[mean_ind]
    for i in range(iterations):
        dists = []
        idx = []
        clusters = []
        for i in range(len(mean_ind)):
            dists.append(euc_norm(X, mean_K[i]))

        dists = np.array(dists).T
        idx =np.arange(0,len(X),1)
        closest = np.argmin(dists, axis=1)

        for i in range(len(mean_ind)):
            clusters.append(np.mean(X[idx[closest==i]], axis = 0))

        mean_K = np.array(clusters)
        y_hat = closest

    return mean_K, y_hat


# img = plt.imread('omg3.png')
# plt.imshow(img)
# x = img.reshape(-1, 4)
# mean_k, y_hat = K_means(x, 4, iterations = 1)
# new_colors = mean_k[y_hat]
# new_colors = new_colors.reshape(img.shape)
# plt.imshow(new_colors)
# # plt.scatter(X[:,0], X[:,1], c = y_hat)
# # plt.scatter(mean_K[:,0], mean_K[:,1], c = 'black')
# # plt.show()

mean_k, y_hat = K_means(X, 3, 1000)

x=X
def DBI(x, y_hat, mean_k):
    variances = []
    for i in range(len(np.unique(y_hat))):
        variances.append(sigma_dbi(x[y_hat == i]))

    variances = np.vstack(variances)
    out = 0
    for i in range(len(mean_k)):
        out += np.max((np.delete(variances, i, axis=0) + variances[i]) / np.sum(
            (np.delete(mean_k, i, axis=0) - mean_k[i]) ** 2, axis=1))
    dbi_test = out / len(np.unique(y_hat))
    return(dbi_test)


def CHI(X, mean_k, y_hat):
    denom = 0
    num = 0
    n = len(y_hat)
    for k in range(len(mean_k)):
        n_k = np.sum(y_hat ==k)
        m_k = mean_k[k]
        m = np.mean(x, axis = 0)

        denom += np.sum((x[y_hat==k] - m_k) ** 2)
        num += n_k * np.sum((m_k - m)**2)

    return (num/denom) * ((n - len(mean_k))/(len(mean_k) - 1))

def BadCHI(X, mean_k, y_hat):
    total_mean = np.mean(X, axis = 0)
    num = 0
    denom = 0
    for i in range(len(mean_k)):
        n_i = len(X[y_hat==i])
        num += n_i * np.sum((mean_k[i] - total_mean)**2, axis = 0)/(K-1)

    inner = []
    for i in range(len(np.unique(y_hat))):
        inner.append((np.sum(euc_norm(x[y_hat == i], np.mean(x[y_hat==i], axis =0))**2))/(len(X) - K))

    denom = sum(inner)


    cal_index = (num/denom)
    return cal_index

cal_index = CHI(X,mean_k, y_hat)

sil = Silhouette(x, mean_k, y_hat)
print(cal_index)
cal_index = []
db_index = []
for i in range(2, 10):
    cal_temp = []
    db_temp = []
    for j in range(15):
        mean_k, y_hat = K_means(X,i)
        cal_temp.append(CHI(X,mean_k,y_hat))
        db_temp.append(DBI(X,y_hat,mean_k))

    cal_index.append(max(cal_temp))
    db_index.append(min(db_temp))
x_axis = np.arange(2,10,1)

fig = plt.figure()
plt.scatter(X[:,0], X[:,1], c = colors)

fig = plt.figure()
plt.plot(x_axis, cal_index)
plt.title('Cal Index')

fig = plt.figure()
plt.plot(x_axis, db_index)
plt.title('DBI')

