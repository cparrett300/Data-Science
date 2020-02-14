import numpy as np
import pandas as pd
import torch
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

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

# # To get TF-IDF you can just use sklearn  called the sklrean.preprocessing.tfidfvectorizor
#
googleWords = gensim.models.KeyedVectors.load_word2vec_format('word2vec_twitter_model.bin', binary=True)



df = pd.read_csv('df_twitter.csv')
word_to_vec_strings = []
df=df.dropna(axis = 0)

df.replace(',', '', inplace = True, regex = True)
df.replace('\'s', '', inplace = True, regex = True)


for i in range(len(df.tweets)):
    word_to_vec_strings.append((df.tweets[0]))

tokenized = []
for sentence in word_to_vec_strings:
    tokenized.append(word_tokenize(sentence))


X = []
for sentence in tokenized:
    x_i = np.zeros((300, len(sentence)))

    for j in range(len(sentence)):
        try:
            x_i[:,j] = googleWords.get_vector(sentence[j])
        except:
            x_i[:, j] = np.zeros((300,0))

    X.append(x_i)


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

class LSTM(torch.nn.Module):
    def __init__(self, size_long, size_short, size_x, size_out):
        super(LSTM, self).__init__()
        self.layer_forget = Layer(size_x + size_short, size_long)
        self.layer_memory = Layer(size_x + size_short, size_long)
        self.layer_tanh = Layer(size_x + size_short, size_long, activation_func = torch.tanh)
        self.layer_recall_sig = Layer(size_x + size_short, size_short)
        self.layer_recall_tanh = Layer(size_long, size_short, activation_func = torch.tanh)
        self.layer_out_init= Layer(size_short, size_x + size_short)
        self.size_short = size_short
        self.size_long = size_long
    def Forward(self, x):
        mem_short = torch.zeros((1, self.size_short))
        mem_long = torch.zeros((1, self.size_long))
        out = []
        for i in range(x.shape[0]):
            z = torch.cat([x[[i], :], mem_short], dim = 1)
            gate_forget = self.layer_forget.Forward(z)
            gate_mem_sig = self.layer_memory.Forward(z)
            gate_mem_tanh = self.layer_tanh.Forward(z)
            gate_recall_sig = self.layer_recall_sig.Forward(z)
            mem_long = mem_long * gate_forget
            gate_mem_mult = gate_mem_sig * gate_mem_tanh
            mem_long = mem_long + gate_mem_mult
            gate_recall_tanh = self.layer_recall_tanh.Forward(mem_long)
            gate_recall_mult = gate_recall_sig * gate_recall_tanh
            mem_short = gate_recall_mult
            new_out = self.layer_out_init.Forward(gate_recall_mult)
            out.append(new_out)
        out = torch.cat(out, dim=0)
        return out

    def Generate(self, start, iterations):
        mem_short = torch.randn((1, self.size_short))
        mem_long = torch.randn((1, self.size_long))
        out = [start]
        for i in range(iterations):
            z = torch.cat([out[i], mem_short], dim=1)
            gate_forget = self.layer_forget.Forward(z)
            gate_mem_sig = self.layer_memory.Forward(z)
            gate_mem_tanh = self.layer_tanh.Forward(z)
            gate_recall_sig = self.layer_recall_sig.Forward(z)
            mem_long = mem_long * gate_forget
            gate_mem_mult = gate_mem_sig * gate_mem_tanh
            mem_long = mem_long + gate_mem_mult
            gate_recall_tanh = self.layer_recall_tanh.Forward(mem_long)
            gate_recall_mult = gate_recall_sig * gate_recall_tanh
            mem_short = gate_recall_mult
            new_out = self.layer_out_init.Forward(gate_recall_mult)
            out.append(new_out)
            if Decode(out[-1]) == '|':
                break
        out = torch.cat(out, dim=0)
        return out

