# 8. RNN
## 8.1 Sequence model
### 8.1.1 Statistical Tools
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


# Text Processing notes
Two strategies
