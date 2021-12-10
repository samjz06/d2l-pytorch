import sys
sys.path.insert(0, '..')

import os
os.chdir(r'G:\RUTGERS\Dive-into-DL-PyTorch_eng')

# import d2l
from d2l.data.base import Vocab
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

def load_data_time_machine(num_examples=10000):
    """Load the time machine data set (available in the English book)."""
    with open('./data/timemachine.txt') as f:
        raw_text = f.read()
    lines = raw_text.split('\n')
    text = ' '.join(' '.join(lines).lower().split())[:num_examples]
    vocab = Vocab(text)
    corpus_indices = [vocab[char] for char in text]
    return corpus_indices, vocab

corpus_indices, vocab = load_data_time_machine()
# corpus_indices: token -> indices
# vocab:
#   idx_to_token: each token in the text
#   token_to_idx: token and its index

