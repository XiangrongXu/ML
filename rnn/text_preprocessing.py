import collections
import re
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
freqs = [freq for _, freq in vocab.token_freqs]

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])
bifreqs = [freq for _, freq in bigram_vocab.token_freqs]

trigram_tokens = [tup for tup in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])
trifreqs = [freq for _, freq in trigram_vocab.token_freqs]

d2l.plot([freqs, bifreqs, trifreqs], xlabel="token: x", ylabel="frequency: n(x)", xscale="log", yscale="log", legend=["unigram", "bigram", "trigram"])
d2l.plt.show()