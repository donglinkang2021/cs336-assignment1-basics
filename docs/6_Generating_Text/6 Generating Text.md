# 6 Generating text

Now that we can train models, the last piece we need is the ability to generate text from our model. Recall that a language model takes in a (possibly batched) integer sequence of length `sequence_length` and produces a matrix of size `(sequence_length × vocab_size)`, where each element of the sequence is a probability distribution predicting the next word after that position. We will now write a few functions to turn this into a sampling scheme for new sequences.

## Softmax

By standard convention, the language model output is the output of the final linear layer (the “logits”) and so we have to turn this into a normalized probability via the softmax operation, which we saw earlier in Eq 10.

## Decoding

To generate text (decode) from our model, we will provide the model with a sequence of prefix tokens (the “prompt”), and ask it to produce a probability distribution over the vocabulary that predicts the next word in the sequence. Then, we will sample from this distribution over the vocabulary items to determine the next output token.

Concretely, one step of the decoding process should take in a sequence $x_{1...t}$ and return a token $x_{t+1}$ via the following equation,

$$
P(x_{t+1} = i | x_{1...t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)} \quad v = \text{TransformerLM}(x_{1...t})_t \in \mathbb{R}^{\text{vocab\_size}}
$$

where `TransformerLM` is our model which takes as input a sequence of `sequence_length` and produces a matrix of size `(sequence_length × vocab_size)`, and we take the last element of this matrix, as we are looking for the next word prediction at the $t$-th position.

This gives us a basic decoder by repeatedly sampling from these one-step conditionals (appending our previously-generated output token to the input of the next decoding timestep) until we generate the end-of-sequence token `<|endoftext|>` (or a user-specified maximum number of tokens to generate).

## Decoder tricks

We will be experimenting with small models, and small models can sometimes generate very low quality texts. Two simple decoder tricks can help fix these issues. First, in **temperature scaling** we modify our softmax with a temperature parameter $\tau$, where the new softmax is

$$
\text{softmax}(v, \tau)_i = \frac{\exp(v_i/\tau)}{\sum_{j=1}^{|\text{vocab\_size}|} \exp(v_j/\tau)}.
\quad (24)
$$

Note how setting $\tau \to 0$ makes it so that the largest element of $v$ dominates, and the output of the softmax becomes a one-hot vector concentrated at this maximal element.

Second, another trick is **nucleus** or **top-p sampling**, where we modify the sampling distribution by truncating low-probability words. Let $q$ be a probability distribution that we get from a (temperature-scaled) softmax of size `vocab_size`. Nucleus sampling with hyperparameter $p$ produces the next token according to the equation

$$
P(x_{t+1} = i|q) =
\begin{cases}
    \frac{q_i}{\sum_{j \in V(p)} q_j} & \text{if } i \in V(p) \\
    0 & \text{otherwise}
\end{cases}
$$

where $V(p)$ is the smallest set of indices such that $\sum_{j \in V(p)} q_j \ge p$. You can compute this quantity easily by first sorting the probability distribution $q$ by magnitude, and selecting the largest vocabulary elements until you reach the target level of $p$.
