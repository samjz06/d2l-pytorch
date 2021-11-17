%matplotlib inline
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython import display # ?
display.set_matplotlib_formats('svg') # ?

import torch
from torch import Tensor
from torchvision import transforms, datasets

data_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0], std=[1])])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)

xcount = torch.ones((10, 28, 28), dtype=torch.float32)
ycount = torch.ones((10), dtype=torch.float32)
##* Note: by inititalizing as ones, this conduct Laplace smoothing also. 

for data, label in mnist_train:
    y = int(label)
    ycount[y] += 1
    xcount[y,:,:] += data.squeeze() # data.shape=(1,28,28); xcount[y,:,:].shape=(28,28)

py = ycount / ycount.sum()
px = xcount / ycount.reshape((10, 1, 1)) # ycount.shape=(10), xcount.shape=(10,28,28), non-broadcastable
##* Note this py, px are also the MLE.


## hotmap count for each class
fig,ax = plt.subplots(1,10,figsize=(10, 10))
for i in range(10):
    ax[i].imshow(xcount[i,:,:].numpy(), cmap='hot')
    ax[i].axes.get_xaxis().set_visible(False)
    ax[i].axes.get_yaxis().set_visible(False)
plt.show()
print('Class probabilities', py)

## give prediction/recognition (as discrimnant task)
data, label = mnist_test[0]
# recall that we classify y as to the class maximize the posterior prob.
# so we need to compute:
# 1) conditional liklihood. p(x|y). log-lik is computed to avoid overflow.
# 2) p(y): this is governed by the py we trained.
logpx = torch.log(px)
logpxneg = torch.log(1-px)
logpy = torch.log(py)

condloglik = data * logpx + (1-data) * logpxneg
condloglik = condloglik.sum((1,2))
bayespost = condloglik + logpy # log[p(x|y)p(y)] i.e. log[p(y|x)]
bayespost -= torch.max(bayespost) # Normalizing to prevent overflow or underflow
post = torch.exp(bayespost)

post /= post.sum() 
#* normalizing the post to make it a distribution.
#* note that: \sum_y {p(x|y)p(y)}=p(x); so this gives posterior distribution
torch.argmax(post) # map prediction
# wrap this as function
del(logpx, logpy, logpxneg, condloglik, bayespost, post)

def bayespost(data, px, py):
    logpx = torch.log(px)
    logpxneg = torch.log(1-px)
    logpy = torch.log(py)

    logpost = (data*logpx + (1-data)*logpxneg).sum((1,2)) + logpy
    logpost -= logpost.max()
    post = logpost.exp()
    
    post /= post.sum()
    return post

post = bayespost(data, px, py)
post.argmax().item()

# now lets try 10 images for testing
fig, ax = plt.subplots(2, 10, figsize=(10, 3))
cnt = 0
for data, label in mnist_test:
    post = bayespost(data, px, py)
    # pic
    ax[1, cnt].bar(range(10), post) # draw the posterior
    ax[1, cnt].axes.get_yaxis().set_visible(False)
    ax[0, cnt].imshow(data.reshape((28,28)), cmap='hot')
    ax[0, cnt].axes.get_xaxis().set_visible(False)
    ax[0, cnt].axes.get_yaxis().set_visible(False)
    cnt += 1

# now calculate the testing error
cnt = 0
err = 0
for data, label in mnist_test:
    cnt += 1
    post = bayespost(data,px,py)
    
    if (post[int(label)] < post.max()): err += 1

print(f'Naive Bayes has error rate = {err/cnt}')
    