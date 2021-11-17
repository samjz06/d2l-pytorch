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

xcount = torch.ones()

mnist_train.type()
