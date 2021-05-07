import collections
import re
from d2l import torch as d2l

# d2l.DATA_HUB["time_machine"] = (d2l.DATA_URL + "timemachine.txt", "090b5e7e70c295757f55df93cb0a180b9691891a")
corpus, vocab = d2l.load_corpus_time_machine()
print(list(vocab.token_to_idx.items()))