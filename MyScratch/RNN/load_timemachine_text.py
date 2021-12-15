import os
os.chdir(r'g:\RUTGERS\Dive-into-DL-PyTorch_eng')

with open(os.getcwd()+r'/data/timemachine.txt', 'r') as f: # load timemachine text
    raw_text = f.read()

type(raw_text) # str
len(raw_text) # 178979
print(raw_text[:100])
print(raw_text[300:500])

lines = raw_text.split('\n') # split by ' '
type(lines) # list
lines[:10] # 

# for a character-level NLP task:
#   join all sentences together into consecutive words, 
#   lower-cases all, 
#   split by space

text = ' '.join(' '.join(lines).lower().split())
print(text[:100])

# next we instantiate text as an object of Vocab class
# Vocab class is defined as to tokenize the raw text.
vocab = Vocab(text)
corpus_indices = [vocab[char] for char in text]

# Combine everything together

def load_data_time_machine(num_examples=10000):
    """Load the time machine data set (available in the English book)."""
    with open('../data/timemachine.txt') as f:
        raw_text = f.read()
    lines = raw_text.split('\n')
    text = ' '.join(' '.join(lines).lower().split())[:num_examples]
    vocab = Vocab(text)
    corpus_indices = [vocab[char] for char in text]
    return corpus_indices, vocab

