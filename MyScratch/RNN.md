8. RNN
# 8.1 Sequence model
## 8.1.1 Statistical Tools
I) Autoregressive models
$P(x_t|x_{t-1},... x_{1})$

two strategies:
1. Auto-regress: $\tau$, $x_t|x_{t-1},... x_{t-\tau}$
2. Latenet auto-regress: summary of past obs, history: $h_t$. Predict $\hat{x}_t$ as well as update $h_t$ with $x_{t-1}$ and $h_{t-1}$.

Question: how to generate training data

II) Markov models
E.g., $\tau=1$, auto-regress = first-order Markov model. 
For discrete value case, dynamic programming used to compute values along the chain.
Control and Reinforcement learning alg use such tools extensively.

III) Causality


# 8.2 Text Processing notes
**Steps include:**
Reading the dataset -> Tokenization -> Vocabulary

**Summary:**
- Text is an important form of sequence data
- To preprocess text, we split text into tokens, build a vocabulary to map token strings into numerical indices, and convert text data into token indices for models to manipulate.

**Exercise:**
Try to find another 3 commonly used methods to tokenize text.

# 8.3 Language Models and the dataset
Sections:
- Learning a Language Model: Sequential fashion; deep -> learning -> is -> fun. prod of prob.
- Markov Models and n-grams: short-term dependency.
- 

**Summary:**
My notes: 
- In summary, language model is to predict the next token based on what have seen so far. Token can be a char, a word, two words, or triple.
- NN language model:
  - data: a length(time-steps) of tokens form sequences, several sequences form a minibatch.
  - minibatch is fed to NN lang model.

Book summary:
- Language models are key to NLP
- n-grams provide a convenient model for dealing with long sequences by truncating the dependence
- Long sequences suffer from the problem that they occur very rarely or never
- Zipf's law governs the word distribution for not only unigrams but also the other n-grams
- There is a lot of structure but not enough freq to deal with infrequent word combinations efficiently via Laplace smoothing
- The main choices for reading long sequences are random sampling and sequential partitioning. The latter can ensure that the subseq from two adjacent minibatches during iteration are adjacent on the original sequence.

# 8.4 RNN
## RNN w. Hidden States
Questions:
1. are $W_{xh}$ of the same size? This is not a Q, we just dicuss a single recurrent layer.
2. what is $X_{t-1}$? Last minibatch. 
3. Then why should one save $H_{t-1}$ among batches?