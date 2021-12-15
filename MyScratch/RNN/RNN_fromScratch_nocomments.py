import os

from torch._C import device

from d2l.train import grad_clipping
os.chdir(r'G:\RUTGERS\Dive-into-DL-PyTorch_eng')

import sys
sys.path.insert(0, os.getcwd())

import d2l
from d2l.data.base import Vocab
import math
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import time

# load corpus and create vocabulary and calculate indices
corpus_indices, vocab = d2l.load_data_time_machine(num_examples=10000,
                                                   actual_path = os.getcwd()+r'/data/timemachine.txt')

def to_onehot(X,size):
    return F.one_hot(X.long().transpose(0,-1), size)

def get_params():
    def _one(shape):
        return torch.Tensor(size=shape).normal_(std=0.01).to(ctx)
        
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
    # Forward computation of rnn layer.
    # 
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
    
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab, ctx):
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

def train_epoch(data_iter, random_data_iter, data_iter_fn, batch_size, num_steps, loss, params, clipping_theta, lr, ):
    if not random_data_iter: 
        state = init_rnn_state(batch_size, num_hiddens, ctx)
    data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
    l_sum, n = 0.0, 0
    for X, Y in data_iter:
        # initiate state accordingly. Either random or from last batch
        if random_data_iter:
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        else:
            for s in state:
                s.detach_() # if consecutive sampling, use last state 
        inputs = to_onehot(X, len(vocab)) # one-hot encoding
        (outputs, state) = rnn(inputs, state, params) # forward computation
        outputs = torch.cat(outputs, dim=0) # outputs shape (num_steps*batch_size, embedded_size)
        y = Y.t().reshape((-1,)) # Y raw shape:(batch_size, num_steps) -> num_steps*batch_size
        l = loss(outputs, y.long()).mean()
        l.backward()
        with torch.no_grad():
            grad_clipping(params, clipping_theta, ctx)
            d2l.sgd(params, lr, 1)
        l_sum += l.item() * y.numel() # y.numel(): total num of ele in y
        n += y.numel()
        
    return math.exp(l_sum/n) # perplexity

def train_and_pred(corpus_indices, random_data_iter, epoch_nums, batch_size, num_steps, ctx, clipping_theta, lr, prefixes, num_chars):
    if random_data_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    
    params = get_params() # initialize params
    loss = nn.CrossEntropyLoss() # loss
    start = time.time()
    for epoch in range(epoch_nums):
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx) # prepare data_iter for current epoch
        perplexity = train_epoch(data_iter, random_data_iter, data_iter_fn, batch_size, num_steps, loss, params, clipping_theta, lr)
        if (epoch+1) % 50 == 0:
            print(f'epoch={epoch+1}, perplexity={perplexity}, time={time.time()-start}')
            start = time.time()
        if (epoch+1) % 100 == 0:
            for prefix in prefixes:
                print(predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab, ctx))

ctx = d2l.try_gpu()
num_inputs, num_hiddens, num_outputs = len(vocab), 512, len(vocab)
num_epochs, num_steps, batch_size = 500, 64, 32
lr, clipping_theta = 1, 1
prefixes = ['traveller']
train_and_pred(corpus_indices, True, num_epochs, batch_size, num_steps, 
               ctx, clipping_theta, lr,
               prefixes, 30)