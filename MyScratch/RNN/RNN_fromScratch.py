import os
os.chdir(r'G:\RUTGERS\Dive-into-DL-PyTorch_eng')

import sys
sys.path.insert(0, os.getcwd())

import d2l
from d2l.data.base import Vocab
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

# load corpus and create vocabulary and calculate indices
corpus_indices, vocab = d2l.load_data_time_machine(num_examples=10000,
                                                   actual_path = os.getcwd()+r'/data/timemachine.txt')
type(corpus_indices)

# One-hot encoding:
#   After tokenization, we have had vocabulary and their corresponding indices
#   That is, e.g., a sequence of characters, 'asdf12[', we could get a vector of their indices from vocab.token_to_idx, e.g. [12, 3, 54, 2, 6, 10]
#   This would be a sample for RNN to be trained on.
#   However, to feed this vector into RNN, we need a encoding strategy, and the resulted tensor will have shape
#   (number of states, batch size, embedded size)
#   
#   One-hot encoding is the simplest solution; obviously, will not be ideal for a large vocabulary

X = torch.randint(0, 20, (5,)) # 5 random integer, like a character sequence
print(X)
F.one_hot(X, len(vocab))

X = torch.randint(0, len(vocab), (2, 5)) # now generate 2 sequence, each has 5 states
print(X)
X = F.one_hot(X, len(vocab))
X = X.permute(1,0,2) 
X.shape # (num_states, batch_size, embedded_size)


def to_onehot(X,size):
    return F.one_hot(X.long().transpose(0,-1), size)

num_inputs, num_hiddens, num_outputs = len(vocab), 512, len(vocab)
ctx = d2l.try_gpu()
print('Using', ctx)

# Create the parameters of the model, initialize them and attach gradients
def get_params():
    def _one(shape):
        return torch.Tensor(size=shape).normal_(std=0.01).to(ctx)
        # x = torch.Tensor(size=shape).normal_(std=0.01)
        # return torch.tensor(x, device=ctx) # use .clone() for safe copy
        # return torch.Tensor(size=shape, device=ctx).normal_(std=0.01) 
        # # torch.Tensor() is subclass of torch.tensor, `device` argument is not supported

    # Hidden layer parameters
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=ctx)
    # Output layer parameters
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=ctx)
    # Attach a gradient
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, ctx):
    return (torch.zeros(size=(batch_size, num_hiddens), device=ctx), )

def rnn(inputs, state, params):
    # Both inputs and outputs are composed of num_steps matrices of the shape
    # (batch_size, len(vocab))
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs: # iterate over the first dim: num_steps
        # hidden state: h_t = \phi(x_t, h_{t-1}, )
        H = torch.tanh(torch.matmul(X.float(), W_xh) + torch.matmul(H.float(), W_hh) + b_h) 
        Y = torch.matmul(H.float(), W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
    # RMQ: 
    # for single element tuple, comma is required
    # (1) wrong, (1,) correct

X = torch.arange(10).reshape((2, 5))
# inputs = to_onehot(X, len(vocab))
# len(inputs), inputs[0].shape
state = init_rnn_state(X.shape[0], num_hiddens, ctx) 
# single layer, len(state)=1; and state[0].shape = (batch_size, num_hidden)
inputs = to_onehot(X.to(ctx), len(vocab))
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(f'length of outputs={len(outputs)} this should be of length "num_step"\n', 
      f'each timestep, output is of shape {outputs[0].shape}, (batch_size, embedded_size)\n',
      f'each timestep, output is of shape {state_new[0].shape}, (batch_size, num_hidden)')

# This function is saved in the d2l package for future use
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab, ctx):
    """[summary]

    Args:
        prefix ([type]): leading character sequence
        num_chars ([type]): num steps to be predict forward
        rnn ([type]): rnn layer, TRAINED 
        params ([type]): params list for hidden and output layer
        init_rnn_state ([type]): method for init params
        num_hiddens ([type]): num of neuron in hidden layer
        vocab ([type]): vocab, used to refer token from idx, or vice versa
        ctx ([type]): device

    Returns:
        [type]: [description]
    """
    
    state = init_rnn_state(1, num_hiddens, ctx) # batch=1: single string pred
    output = [vocab[prefix[0]]] # initiate output with the index of the first character
    for t in range(num_chars + len(prefix) - 1): 
        # Question:
        #   As output, (H,) is returned by rnn, so H is the last hidden state
        #   then why shouldnt we start with the last char of prefix?
        # Answer: 
        #   Because you are making prediction on a TESTING data...so cal state from the begining
        
        # The output of the previous time step is taken as the input of the
        # current time step.
        # one char each time
        X = to_onehot(torch.tensor([output[-1]],device=ctx), len(vocab)) 
        # Calculate the output and update the hidden state
        (Y, state) = rnn(X, state, params)
        # The input to the next time step is the character in the prefix or
        # the current best predicted character
        if t < len(prefix) - 1:
            # Read off from the given sequence of characters
            output.append(vocab[prefix[t + 1]])
        else:
            # This is maximum likelihood decoding. Modify this if you want
            # use sampling, beam search or beam sampling for better sequences.
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([vocab.idx_to_token[i] for i in output])

# RMQ:
# transpose() can only swap two dimensions, permute() can swap all dim(has to provide all)
# if you need a copy use clone();
# if you need the same storage use view().
# The semantics of reshape() are that it may or may not share the storage and you don’t know beforehand.
