import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def euclidean_norm(x, y, *args):
    return np.sum(((x - y) ** args[0]), axis =1)

class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def D(self, z):
        return self(z) * (1 - self(z))

class softMax:
    def __init__(self):
        pass
    def __call__(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
#def softMax(z):
#        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

class Tanh:
    def __init__(self):
        pass
    def __call__(self, z):
        return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
    def D(self,z):
        return 1 - self(z) ** 2


# SIMULATING DATA
'''      
N = 10000
K = 5
D = 2
X0 = np.random.randn((N//K), D)/4 + np.array([4,4])
X1 = np.random.randn((N//K),D)/4 + np.array([0,-2])
X2 = np.random.randn((N//K),D)/4 + np.array([-4,-0])
X3 = (np.random.randn((N//K),D)/4 + np.array([-4,-2]))
X4 = (np.random.randn((N//K),D)/4 + np.array([4,-4]))

X = np.vstack((X0, X1, X2, X3, X4))

x_max = np.max(X, axis = 0)
x_min = np.min(X, axis = 0)

X = (X - x_min)/x_max
y = np.array([0]*(N//K) + [1] * (N//K) + [2] * (N//K) + [3] * (N//K) + [4] * (N//K))
y0 = y==0
y1 = y==1
y2 = y==2
y3 = y==3
y4 = y==4

y = np.column_stack((y0, y1, y2, y3, y4))
  '''
plt.close('all')
N = 10000
K = 2
X0 = np.random.randn(N//K, 2)/8 #center
X1 = (np.random.rand(N*5, 2) - .5) * 4
x_col0 = X1[:,0]
x_col1 = X1[:,1]
x_col0 = x_col0[((X1[:,0] ** 2) + (X1[:,1] ** 2)) > .8]
x_col1 = x_col1[((X1[:,1] ** 2) + (X1[:,0] ** 2)) > .8]

X2 = np.column_stack((x_col0[:X0.shape[0]], x_col1[:X0.shape[0]]))
y = np.array([0]*(N//K) + [1] * (N//K))
y0 = y==0
y1 = y==1
y = np.column_stack((y0, y1))



fig = plt.figure()
plt.scatter(X0[:,0], X0[:,1], c='green')
plt.scatter(X2[:,0], X2[:,1], c='red')
X = np.vstack((X2, X0))
# Defining neural network
z_size = 6
sig = Sigmoid()

w_1 = np.random.randn(z_size, y.shape[1])
b_1 = np.random.randn(1, y.shape[1])

w_0 = np.random.randn(X.shape[1], z_size)
b_0 = np.random.randn(1, z_size)

epochs = 1000
learn_rate = 1e-3
losses = []
for i in range(epochs):
    z = sig(X @ w_0 + b_0)
    p_hat = softMax(z @ w_1 + b_1)
    grad_p_hat = p_hat - y
    grad_w_1 = z.T @ grad_p_hat
    grad_b_1 = np.sum(grad_p_hat, axis =0, keepdims = True)
    grad_w_0 = X.T @ (grad_p_hat @ w_1.T * sig.D(X @ w_0 + b_0))
    grad_b_0 = np.sum(grad_p_hat @ w_1.T * sig.D(X @ w_0 + b_0), axis = 0, keepdims = True)

    w_1 -= grad_w_1 * learn_rate
    w_0 -= grad_w_0 * learn_rate
    b_1 -= grad_b_1 * learn_rate
    b_0 -= grad_b_0 * learn_rate
    loss = -np.sum(y * np.log(p_hat + 1e-50))
    losses.append(loss)

fig = plt.figure()
plt.plot(losses)
plt.show()

def XLine(x):
    x_max = np.max(x, axis = 0)
    x_min = np.min(x, axis = 0)
    x_line = np.random.rand(10000, 2) * (x_max - x_min) + x_min
    return x_line
x_line = XLine(X)

z_line = sig(x_line @ w_0 + b_0)

y_line = z_line @ w_1 + b_1



fig = plt.figure()
plt.scatter(x_line[:,0], x_line[:,1], c=np.argmax(y_line, axis =1), cmap ='jet')#, alpha = 0.2, s = 3)
plt.scatter(X[:,0], X[:,1], c = np.argmax(p_hat, axis = 1))