import pandas as pd
import re

"""
Dataset-specific preprocessing function.

This function was tailored to the current dataset and may not
generalize well to others. Regular expressions are used here
mainly for convenience, not necessity.
"""

def process(root):
    sentences = pd.read_csv(root)

    def preprocess(sentence):
        sentence = sentence.lower().strip()
        sentence = re.sub(r"[^a-zA-Z\u0900-\u097F0-9?.!,¿]+", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence

    sentences = sentences.dropna(subset = ['english', 'hindi'])
    sentences['english'] = sentences['english'].apply(preprocess)
    sentences['hindi'] = sentences['hindi'].apply(lambda x: '<sos> ' + preprocess(x) + ' <eos>')
    sentences = sentences[(sentences['english'].str.split().str.len() > 5) & (sentences['english'].str.split().str.len() <= 30)]
    sentences = sentences[(sentences['hindi'].str.split().str.len() > 5) & (sentences['hindi'].str.split().str.len() <= 30)]

    sentences = sentences[:20000]
    return sentences