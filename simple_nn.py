import pandas as pandas
import numpy as np
import torch 
import torch.nn as nn
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.piecewise(
        x,
        [x > 0],
        [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
    )


def softmax(x):
    '''
    preds should be sigmoid(input)
    '''
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def del_cross_entropy_loss(preds, target):
    '''
    preds should be sigmoid(input)
    '''
    return preds - target

def forward(layers, input):
    acts = [input]
    for layer in layers:
        acts = acts + [layer(acts[-1])]
    return acts

def backward(acts, targs, layers, lr):
    total_grad = 0
    output_layer = True
    for i, layer in enumerate(layers[::-1]):
        curr_output = acts[-(i + 1)]
        curr_input = acts[-(i + 2)]
        if output_layer:
            output_layer = False
            curr_grad = layer.grad_input(curr_output, targs)
        else:
            curr_grad = layer.grad_input(curr_output, total_grad)
        #Update current layer with gradient accumulated up to this point
        layer_grad = layer.grad_self(curr_input, total_grad)
        if layer_grad is not None:
            #Update the current layer
            layer.update_grad(lr * layer_grad)
        #Set accumulated gradient to include this layer
        total_grad = curr_grad

class Linear():
    def __init__(self, dim_in, dim_out):
        self.weights = np.random.normal(0, 1, size=[dim_in, dim_out])
    
    def __call__(self, x):
        return x @ self.weights
    
    def grad_self(self, input, total_grad):
        return input.T @ total_grad
    
    def grad_input(self, output, total_grad):
        return total_grad @ self.weights.T
    
    def update_grad(self, grad):
        self.weights = self.weights - grad

class Sigmoid():
    def __call__(self, x):
        return sigmoid(x)
    
    def grad_input(self, output, grad):
        sig_out = sigmoid(output)
        grad_sigmoid = np.multiply(sig_out, (1 - sig_out))
        return np.multiply(grad_sigmoid, grad)
    
    def update_grad(self, grad):
        pass
    
    def grad_self(self, input, total_grad):
        pass

class Softmax():
    def __call__(self, x):
        return softmax(x)
    
    def grad_input(self, output, target):
        return (output - target)/output.shape[0]
    
    def update_grad(self, grad):
        pass
    
    def grad_self(self, input, total_grad):
        pass

class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, preds, target):
        #print(target, preds)
        return  -np.multiply(target, np.log(preds)).sum() / preds.shape[0]

mnist = load_digits()
data = mnist['data']
targs = mnist['target']
labels = np.zeros((targs.shape[0], 10))
labels[np.arange(len(labels)), targs] += 1


train_size = int(len(data)*.9)
test_size = len(data) - train_size
train_inds = np.random.choice(list(range(train_size)), size=train_size)
test_inds = list(set(list(range(len(data)))) - set(train_inds))

train_set = data[train_inds]
train_labels = labels[train_inds]

test_set = data[test_inds]
test_labels = labels[test_inds]


def train(model, lr, loss_func, bs, epochs, train_data, train_labels, valid_data, valid_labels):
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        n_batches = (len(train_data)//bs) + 1
        all_batch_inds = list(range(len(train_data)))
        np.random.shuffle(all_batch_inds)
        epoch_train_loss = []
        epoch_valid_loss = []
        for batch in range(n_batches):
            batch_inds = all_batch_inds[batch * bs: min((batch + 1) * bs, len(all_batch_inds))]
            batch_data = train_data[batch_inds]
            batch_labels = train_labels[batch_inds]
            acts = forward(model, batch_data)
            loss = loss_func(acts[-1], batch_labels)
            epoch_train_loss += [loss]
            backward(acts, batch_labels, model, lr)
        epoch_train_loss = np.array(epoch_train_loss).mean()
        valid_acts = forward(model, valid_data)
        epoch_valid_loss = loss_func(valid_acts[-1], valid_labels)
        train_loss += [epoch_train_loss]
        valid_loss += [epoch_valid_loss]
        print(f"Epoch {epoch} train loss: {epoch_train_loss}, valid loss {epoch_valid_loss}")
    return train_loss, valid_loss


loss_func = CrossEntropyLoss()
model = [
    Linear(train_set[0].shape[0], 300),
    Sigmoid(),
    Linear(300, 200),
    Sigmoid(),
    Linear(200, 10),
    Softmax()
]

lr = 0.005
bs = 32
epochs = 35

train_loss, valid_loss = train(model, lr, loss_func, bs, epochs, train_set, train_labels, test_set, test_labels)

def plot_loss(fn, train_loss, valid_loss):
    plt.axis('square')
    plt.figure(figsize=(18,18))
    plt.rcParams.update({'font.size': 22})
    plt.plot([i for i, x in enumerate(train_loss)], train_loss, label="train loss")
    plt.plot([i for i, x in enumerate(valid_loss)], valid_loss, label="valid loss")
    plt.legend()
    plt.ylim(0, max(max(train_loss), max(valid_loss)))
    plt.xlim(0, len(train_loss))
    plt.savefig(fn, bbox_inches='tight')
    plt.close()
    
plot_loss("simple_nn_loss", train_loss, valid_loss)

    
#Pytorch version
def train_one_epoch(model, optimizer, loss_func, training_loader, validation_loader):
    train_losses = []
    model.train(True)
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses += [loss.item()]
    epoch_train_loss = np.array(train_losses).mean()
    model.train(False)
    valid_losses = []
    for i, data in enumerate(validation_loader):
        inputs, labels = data
        outputs = model(inputs)
        valid_losses += [loss_func(outputs, labels).item()]
    epoch_valid_loss = np.array(valid_losses).mean()
    return epoch_train_loss, epoch_valid_loss


def train_model(model, fn):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        epoch_train_loss, epoch_valid_loss = train_one_epoch(model, optimizer, loss_func, training_loader, validation_loader)
        train_losses += [epoch_train_loss]
        valid_losses += [epoch_valid_loss]
        print(f"{epoch}:{round(epoch_valid_loss, 5)}\\\\")
    plot_loss(fn, train_losses, valid_losses)


def init_kaiming(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_ones(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -1, 1)
        m.bias.data.fill_(0.01)

 
def init_zero(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0.01)


class SimpleNN(nn.Module):
    def __init__(
        self,
        init="xavier_uniform",
    ):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(train_set[0].shape[0], 300),
            nn.Sigmoid(),
            nn.Linear(300, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Softmax()
        )
        if init == "xavier_uniform":
            self.layers.apply(init_kaiming)
        elif init == "rand_1":
            self.layers.apply(init_ones)
        else:
            self.layers.apply(init_zero)
        
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input


loss_func = nn.CrossEntropyLoss()

lr = 0.05
bs = 8
epochs = 35
train_data = []
for i in range(len(train_set)):
   train_data.append([torch.tensor(train_set[i]).to(dtype=torch.float32), train_labels[i]])

test_data = []
for i in range(len(test_set)):
   test_data.append([torch.tensor(test_set[i]).to(dtype=torch.float32), test_labels[i]])


training_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
validation_loader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=False)


model_params = [("zero", "init_zero_loss.png"), 
                ("rand_1", "init_ones_loss.png"), 
                ("xavier_uniform", "init_kaiming_loss.png")]


for params in model_params:
    train_model(SimpleNN(params[0]), params[1])

