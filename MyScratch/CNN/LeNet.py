import torch
import torch.nn as nn
from torchsummary import summary


# LeNet: 
# Complicated way of constructing LeNet
class Reshape(nn.Module):
    # note: 
    # construct every layer inheriting nn.Module so that it could be export by summary method from torchsummary
    def forward(self, x):
        return x.view(-1,1,28,28)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # conv
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), padding=2)    
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5))
        # linear
        self.linear1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
        # actv fun
        self.sig = nn.Sigmoid()
        # pooling
        self.avgp = nn.AvgPool2d(kernel_size=2, stride=2)
        # flatten
        self.flatten = nn.Flatten()
        # reshape
        self.reshape = Reshape() #! cannot Reshape(x), as it = Reshape.__init__(x).
        
    def forward(self, x):
        # reshape
        # x = x.view(-1, 1, 28, 28) # otherwise, how do call function inside this
        x = self.reshape(x)
        # conv layer 1
        x = self.avgp(self.sig(self.conv1(x)))
        # conv layer 2
        x = self.avgp(self.sig(self.conv2(x)))
        # flatten for linear
        # x = x.view(x.shape[0], -1)
        x = self.flatten(x)
        # linear 1
        x = self.sig(self.linear1(x))
        # linear 2
        x = self.sig(self.linear2(x))
        # linear 3
        x = self.linear3(x)
        
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = LeNet().to(device) # torchsummary by default is run on GPU if available
summary(net, (1,28,28))

# Alternatively, construct network as torch.nn.Sequential() class    
d2lnet = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=16*5*5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

X = torch.randn(size=(1,1,28,28), dtype = torch.float32)
for l in d2lnet:
    X = l(X)
    print(l.__class__.__name__)
    
    
# training LeNet with Fashion-MNIST