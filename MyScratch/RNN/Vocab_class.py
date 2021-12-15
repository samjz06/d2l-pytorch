import os
os.chdir(r'g:\RUTGERS\Dive-into-DL-PyTorch_eng')

import torch
import random
from collections import Counter

# with open(os.getcwd()+r'/data/timemachine.txt', 'r') as f: # load timemachine text
with open('../data/timemachine.txt', 'r') as f:
    raw_text = f.read()


class Vocab:
    """
        Vocabulary for text.
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        # sort according to frequencies
        counter = self.count_corpus(tokens) #! returns a dict, I think
        self._token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True) #! in alphabetic order
        # The index for the unknown token is 0
        # 'self.idx_to_token' list, records all tokens
        # 'self.token_to_idx' dict, records token and its corresponding idx
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)
        }
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break #! break is ok since freq is sorted
            if token not in self.token_to_idx:
                self.idx_to_to_token.append(token) # add this to dict
                self.token_to_idx[token] = len(self.idx_to_token)-1 # a new number(idx)
            # else:
            #     pass
            # # do nothing
    
    def __len__(self):
        return len(self.idx_to_token) 
    #? why is this used: 
    # So that can call len(object)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property #TODO: Figure out the use of @property decorator
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs
    
    def count_corpus(tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return Counter(tokens)
    
# RMQ:
# Alternatively, another way of writing this Vocab
class Vocab(object):  # This class is saved in d2l.
    """[summary]
        By instantiating an object, Vocab create a vocabulary(character-level) based on given corpus
    """
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        # sort by frequency and token
        counter = collections.Counter(tokens)
        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True) # sort tokens by freq descendingly
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        else:
            self.unk = 0
            tokens = ['<unk>']
        tokens +=  [token for token, freq in token_freqs if freq >= min_freq]
        self.idx_to_token = []
        self.token_to_idx = dict()
        for token in tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
        # self.idx_to_token: List, append token(letter) when see it for the first time
        # self.token_to_idx: dict, token: index, e.g. 'a': 1. Think as 'Letter ID'.

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk) 
            # dict.get(key, value): 
            # key: key to be searched
            # value: value to be returned if key not founded
        else:
            return [self.__getitem__(token) for token in tokens] # if require multiple tokens

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)): # when inquire about single token
            return self.idx_to_token[indices] # return corresponding index
        else:
            return [self.idx_to_token[index] for index in indices]
        

if __name__ == '__main__':
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])
