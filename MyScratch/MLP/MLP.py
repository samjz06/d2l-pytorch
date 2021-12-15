import os
import sys
sys.path.insert(0, '..') # insert path '..' at the 0th position
# import d2l
# from d2l.data import load_data_fashion_mnist
# from d2l.train import sgd, train_ch3


import torch
import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader


# my fancy MLP
class Net(nn.Module): # inherit from nn.Module
    def __init__(self, num_inputs=784, num_outputs=10, num_hiddens=256, is_training=True):
        super(Net, self).__init__()
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        
        self.linear_1 = nn.Linear(num_inputs, num_hiddens)
        self.linear_2 = nn.Linear(num_hiddens, num_outputs)
        
        self.relu = nn.ReLU()
        
    def forward(self, X):
        X = X.reshape((-1, self.num_inputs)) # n * M: M is the number of features/pixels
        H1 = self.relu(self.linear_1(X)) # net activation value
        out = self.linear_2(H1)
        
        return out
    
net = Net()
print(net)

# then implement this network to classify fashion-mnist
## load dataset:
def load_data_fashion_mnist(batch_size=256, num_workers=4, resize=None, root='./data/'):
    trans = []
    if resize:
        trans += [transforms.Resize(resize)]
    trans += [transforms.ToTensor()]
    trans = transforms.Compose(trans)
    
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=trans, target_transform=None, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, transform=trans, target_transform=None, download=True)
    
    # use DataLoader to load minibatch
    if sys.platform.startswith('win'):
        num_workers=0
    else:
        num_workers=num_workers

    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size, shuffle=True, num_workers=num_workers)
    
    return train_iter, test_iter

def evaluate_accuracy(test_iter, net, device=torch.device('cpu')):
    net.eval()
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in test_iter:
        # Copy data to device
        X, y = X.to(device), y.to(device) #? pytorch .to()
        with torch.no_grad(): #? Y
            y = y.long() #? why long type
            acc_sum += (net(X).argmax(dim=1) == y).sum()
            n += y.size()[0]
    return acc_sum.item()/n

def train(net, train_iter, test_iter, loss, num_epochs, batch_size, lr):
    optimizer = optim.SGD(net.parameters(), lr=lr) # use SGD
    
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0 # an new epoch=an new sweep over data with batch_size
        for X, y in train_iter:
            optimizer.zero_grad() # initialize grad vector
            
            yhat = net(X) # train fit (under current parameters)
            train_loss = loss(yhat, y) # cross entropy
            train_loss.backward() # compute gradient
            optimizer.step() # execute GD; update parameters
            
            y = y.type(torch.float32) #? why no long type now
            train_loss_sum += train_loss.item()
            train_acc_sum += (yhat.argmax(dim=1).type(torch.FloatTensor)==y).detach().sum().float() #? Y type change and detach
            n += y.size()[0] # this n is to track number of training since data could be load parallel
        # see how this epoch performs
        test_acc = evaluate_accuracy(test_iter, net)
        print(f'epoch {epoch+1}, loss {train_loss_sum/n}, train acc {train_acc_sum/n}, test acc {test_acc}')

train_iter, test_iter = load_data_fashion_mnist()
loss = nn.CrossEntropyLoss() # loss function
num_epochs,batch_size,lr = 10,256,0.5

train(net, train_iter, test_iter, loss, num_epochs, batch_size, lr)