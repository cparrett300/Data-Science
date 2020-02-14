import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from softMax import softMax

def Sigmoid(z):
    return 1/(1+ np.exp(-z))
D = 2
K = 2
N = int(K * 1e3)

X0 = np.random.randn((N//K), D) + np.array([2,2])
X1 = np.random.randn((N//K),D) + np.array([0,-2])
X2 = np.random.randn((N//K),D) + np.array([-2,-0])
X = np.vstack((X0, X1, X2))
y = np.array([0]*(N//K) + [1] *(N//K) + [2] * (N//K))
y = y.reshape(-1, 1)
#ax1 = fig.add_subplot(111)

weights = np.random.randn(X.shape[1], y.shape[1])
bias = np.random.randn(1, y.shape[1])

iterations = 1000
learn_rate = 1e-3
losses = []
for i in range(iterations):
    y_hat = X @ weights + bias
    p_hat = softMax(y_hat)
    loss = np.sum(y * np.log(p_hat + 1e-50) + (1 - y) * np.log(1 - p_hat + 1e-50))
    losses.append(loss)
    weights = weights + learn_rate * X.T @ (p_hat - y)
    bias = bias + learn_rate * np.sum(p_hat - y, axis = 0)

y_hat = X @ weights + bias
p_hat = Sigmoid(y_hat)




class BinaryLogisticRegression:
    def __init__(self, shape_in):
        self.weights = np.random.randn(shape_in, 1)
        self.bias = np.random.randn(1, 1)

    def fit(self, X, y, learn_rate = 1e-3, iterations = 1000):
        losses = []
        for i in range(iterations):
            y_hat = X @ self.weights + self.bias
            p_hat = softMax(y_hat)
            loss = np.sum(y * np.log(p_hat + 1e-50)))
            losses.append(loss)
            self.weights = self.weights - learn_rate * X.T @ (p_hat - y)
            self.bias = self.bias - learn_rate * np.sum(p_hat - y, axis=0)

    def predict(self, X):
        y_hat = X @ self.weights + self.bias
        p_hat = Sigmoid(y_hat)
        return(y_hat, p_hat)

model = BinaryLogisticRegression(X.shape[1])
model.fit(X,y)
y_hat, p_hat = model.predict(X)

threshhold_vec = np.arange(0, 1, step=.005)

def bestThreshhold(p_hat, y, threshhold_vec):
    results_names = ['TPR', 'FPR', 'Precision', 'Recall', 'F1']
    results = np.zeros([len(threshhold_vec), 5])
    for thresh_ind in range(len(threshhold_vec)):
        threshhold = threshhold_vec[thresh_ind]
        positive = (p_hat > threshhold)
        negative = (p_hat <= threshhold)
        TP = np.sum(positive[positive] == y[positive])
        FP = len(positive[positive]) - TP

        FN = np.sum(negative[negative] == y[negative])
        TN = len(negative[negative]) - FN

        TPR = TP/(TP + FN)
        FPR = FP/(FP + TN)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        F1 = (2 * precision * recall)/ (precision + recall)
        results[thresh_ind, 0] = TPR
        results[thresh_ind, 1] = FPR
        results[thresh_ind, 2] = precision
        results[thresh_ind, 3] = recall
        results[thresh_ind, 4] = F1

    return(threshhold_vec[np.argmax(results, axis=0)], results_names, results)

best_thresh, thresh_names, threshhold_mat = bestThreshhold(p_hat, y, threshhold_vec)

#alternative y
fig = plt.figure()
plt.scatter(X[:,0], X[:,1], c = p_hat.flatten())
plt.show()
fig2 = plt.figure()
plt.plot(losses)
plt.show()

for i in range(len(thresh_names)):
    fig_i = plt.figure()
    plt.scatter(threshhold_vec, threshhold_mat[:,i])
    plt.title(thresh_names[i])
    plt.show()

fig_roc = plt.figure()
plt.scatter(threshhold_mat[:,1], threshhold_mat[:,0])
plt.plot([0, 1], [0,1])
plt.title('ROC Curve')