import torch
from matplotlib import pyplot as plt
x = (torch.rand(1000, 1) - .5)* 10
y = x **2 + torch.randn(x.shape)*10

plt.scatter(x,y)

x = torch.cat([x, y], dim=1)
def Nothing(x):
    return x

class Layer(torch.nn.Module):
    def __init__(self, size_in, size_out, activation_func=Nothing):
        super(Layer, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(size_in, size_out, requires_grad=True))
        self.b = torch.nn.Parameter(torch.randn(1, size_out, requires_grad=True))
        self.activation_func = activation_func

    def Forward(self, x, noise_rate=0, threshhold=0):
        x = self.activation_func(x @ self.w + self.b)
        noise = torch.randn(x.shape) * noise_rate
        mask = torch.tensor(np.random.uniform(0, 1, x.shape)) > threshhold
        x = (x + noise) * mask
        return (x)

    def Predict(self, x):
        return (self.activation_func(x @ self.w + self.b))

class AutoEncoder(torch.nn.Module):
    def __init__(self, size_in, size_reduced):
        super(AutoEncoder, self).__init__()
        self.layer_down = Layer(size_in, size_reduced)
        self.layer_up = Layer(size_reduced, size_in, lambda x: x)

    def Forward(self, x):
        return self.layer_up.Forward(self.layer_down(x))

    def Reduce(self, x):
        return self.layer_down(x)

model = AutoEncoder(2,1)
iterations = 1000
opt = torch.optim.Adam(model.parameters())
loss_func = torch.nn.MSELoss()
for i in range(iterations):
        x_hat = model.Forward(x)
        loss = loss_func(x_hat, x)
        loss.backward() # saves loss within whatever requires gradient
        opt.step()
        opt.zero_grad()
