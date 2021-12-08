import sys
sys.path.insert(0, '..')

import torch
import random
import collections

with open('../data/timemachine.txt', 'r') as f:
    raw_text = f.read()
print(raw_text[0:110])

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
        counter = count_corpus(tokens) #! returns a dict, I think
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
        return len(self.idx_to_token) #? why is this used
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs
    
    def count_corpus(tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)
    
# usage
if __name__ == '__main__':
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])
