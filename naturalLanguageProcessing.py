from matplotlib import pyplot as plt

import nltk
from nltk.corpus import stopwords
# # nltk.download('wordnet')
# # nltk.download('punkt')
# # nltk.download('averaged_perceptron_tagger')
# # nltk.download('stopwords')
# stopwords.words('english')
#
#
# stemmer = nltk.stem.PorterStemmer()
#
# lemmatizer= nltk.stem.WordNetLemmatizer()
# lemmatizer.lemmatize(('jogging', pos = 'v'))
# stringStuff = "That wasn't very cash money of you, was it? " \
# "fdsafdasd" \
# "cool beans"
# tokens = nltk.tokenize.word_tokenize(stringStuff)
# tokenizer = nltk.tokenize.SyllableTokenizer()
# tokenizer.tokenize()
#
# nltk.ngrams(tokens)
# np.unique(tokens, return_counts = True)

import torch
import gensim
import numpy as np

# # To get TF-IDF you can just use sklearn like a dumbass, called the sklrean.preprocessing.tfidfvectorizor
#
# googleWords = gensim.models.keyedvectors._load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True

x = torch.linspace(-1 *np.pi, 10 * np.pi, steps = 1000)
x = x.reshape(-1, 1)
y = torch.sin(x)
y = y.reshape(-1, 1)
i = 0
memory = torch.zeros(1,5)
torch.cat([x[[i], :], memory], dim = 1)

class Layer(torch.nn.Module):
    def __init__(self, size_in, size_out, activation_func=torch.sigmoid):
        super(Layer, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(size_in, size_out, requires_grad=True))
        self.b = torch.nn.Parameter(torch.randn(1, size_out, requires_grad=True))
        self.activation_func = activation_func

    def Forward(self, x):
        return self.activation_func(x @ self.w + self.b)


class RNN(torch.nn.Module):
    def __init__(self, size_in, size_out, size_mem):
        super(RNN, self).__init__()
        self.layer_0 = Layer(size_in + size_mem, size_mem)
        self.layer_out = Layer(size_mem, size_out)
        self.size_mem = size_mem
    def Forward(self, x):
        mem = torch.zeros((1, self.size_mem))
        out = []
        for i in range(x.shape[0]):
            z = torch.cat([x[[i], :], mem], dim = 1)
            mem = self.layer_0.Forward(z)
            out.append(self.layer_out.Forward(mem))

        out = torch.cat(out, dim=0)
        return out
    def Generate(self, start, iterations):
        mem = torch.randn((1, self.size_mem))
        out = [start]
        for i in range(iterations):
            z = torch.cat([out[i], mem], dim = 1)
            mem = self.layer_0.Forward(z)
            out.append(self.layer_out.Forward(mem))

        out = torch.cat(out, dim=0)
        return out


model = RNN(1, 1, 5)
iterations = 100
loss_func = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
for i in range(iterations):
    y_hat = model.Forward(x)
    loss = loss_func(y_hat[:-1], y[1:])
    loss.backward()
    opt.step()
    opt.zero_grad()
    losses.append(loss.detach)


fig = plt.figure()
plt.plot(x[1:], y[1:], c = 'blue')
plt.plot(x[1:], y_hat[:-1].detach(), c= 'red')

# words = torch.ones([1, 300])
# memory = torch.zeros([1, 50])
# torch.cat(words, memory, dim = 1)


y_gen = model.Generate(y[[0], :], 1000)

plt.plot(x[1:], y_gen[1:-1].detach())
#
# w_mem = torch.randn(size_in, size_mem)
# w_out = torch.randn(size_in, size_out)
# self.b_out = torch.randn(1, size_out)
# self.b_mem = torch.randn(1, size_mem)
# for i in x:
#     j = torch.cat([i, memory], dim=0)
#     memory = j @ w_mem + b_mem
#     y_hat = j @ w_out + b_out

