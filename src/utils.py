import numpy as np
import pandas as pd
from collections import Counter

def find_threshold(sentences, column = "hindi"):
    max_sequence_length = 0.0
    for sentence in sentences[column]:
        current_len = len(str(sentence).split())
        max_sequence_length = max(max_sequence_length, current_len)
    return max_sequence_length

def create_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        sentence = str(sentence)
        counter.update(sentence.split())

    vocab = {word: i + 4 for i, word in enumerate(counter.keys())}
    vocab["<pad>"] = 0
    vocab["<sos>"] = 1
    vocab["<eos>"] = 2
    vocab["<unk>"] = 3
    return vocab

def tokenize(sentence, vocab):
    tokens = [vocab.get(word, vocab["<unk>"]) for word in str(sentence).split()]
    return tokens

def pad_sequence(tokens, limit):
    padded_sequence = [0] * (limit - len(tokens))
    padded_sequence = np.array(tokens + padded_sequence)
    return padded_sequence