import torch
import torch.nn as nn

# for verification
import sys
sys.path.insert(0, '..')
import d2l

# Convolution computation(Cross-correlation)
def corr2d(X, K):
    '''
    X: 2d tensor, stands for image(single-channel)
    K: 2d tensor, convolution kernel
    '''
    h, w = X.shape
    k_h, k_w = K.shape
    Y = torch.zeros((h-k_h+1, w-k_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):    
            Y[i, j] = torch.sum(X[i:(i+k_h), j:(j+k_w)] * K)
    
    return Y

## check my code with tutorial d2l.corr2d
X = torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.Tensor([[0, 1], [2, 3]])
print(corr2d(X, K)==d2l.corr2d(X, K)) # pass
# del(X,K)

## multi input channel
def corr2d_multi_in(X, K):
    '''
        multiple input channel
        
        kernel shape: c_i * k_h * k_w
    '''
    return torch.sum([corr2d(x,k) for x,k in zip(X,K)])
## multi in and out channel
def corr2d_multi_inout(X, K):
    '''
        convolution kernel shape: is c_o * c_i * k_h * k_w
    '''
    return torch.stack([corr2d_multi_in(X, k) for k in K])

## Convolutional layers (inherit from nn.Module class)
class Conv2D(nn.Module):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(*kwargs)
        self.weight = torch.randn(kernel_size, dtype=torch.float32, requires_grad=True)
        self.bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias



## use PyTorch embedded function
### padding and stride are also included
nn.Conv2d()
# rmq:
# kernel_size: by default will be the same for height and width
# padding is for !!both!! sides

### pooling
nn.MaxPool2d(kernel_size=(3,3), padding=1, stride=2)

# shape of output is: floor((n+p-k+s)/s)
m = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,2), stride=1, padding=0)
m.weight.shape # (n_bat,n_ch,h,w)
m.weight = nn.Parameter(torch.Tensor([[0, 1], [2, 3]]).reshape((1,1)+m.weight.shape), requires_grad=True)
m(X.reshape((1,1)+X.shape))
## check the result again

    
    










