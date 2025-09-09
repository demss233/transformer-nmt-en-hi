from richprint import RichPrint
rprint = RichPrint()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import datasets, models
from torch.utils.data import DataLoader

import os
import re
import sys
import math
import random
from tqdm.auto import tqdm
from collections import Counter

from .data import process
from .utils import *
from .torch_data import initialize
from .transformer import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Working on the [device]:", device)

root = '/kaggle/input/hindi-english-parallel-corpus/hindi_english_parallel.csv'
sentences = process(root)

source_vocab = create_vocab(sentences["english"])
target_vocab = create_vocab(sentences["hindi"])

dataloader = initialize(
    source_vocab = source_vocab,
    target_vocab = target_vocab,
    sentences = sentences,
    device = device
)

embedding_dim = 256
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 512
dropout = 0.1
pad_idx = 0
learning_rate = 1e-4
epochs = 20
source_vocab_size = len(source_vocab)
target_vocab_size = max(target_vocab.values()) + 1
hidden_dim = 512
teacher_forcing_ratio = 0.5
max_sequence_length = 40

model = Seq2Seq(
    num_encoder_layers,
    num_decoder_layers,
    embedding_dim,
    nhead,
    source_vocab_size,
    target_vocab_size,
    dim_feedforward,
    dropout,
    pad_idx
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = learning_rate,
    betas = (0.9, 0.98),
    eps = 1e-9
)

losses = []

for epoch in range(epochs):
    running_loss = []
    model.train()
    progress_bar = tqdm(dataloader, desc = f"Epoch {epoch + 1}/{epochs}", leave = False)

    for source, target in progress_bar:
        optimizer.zero_grad()
        outputs = model(source, target[:, :-1])
        outputs = outputs.reshape(-1, outputs.shape[2])

        target = target[:, 1:].reshape(-1)
        loss = criterion(outputs, target)

        running_loss.append(loss.item())
        progress_bar.set_postfix({"Loss": loss.item()})

        loss.backward()
        optimizer.step()

    avg_loss = sum(running_loss) / len(running_loss)
    losses.append(avg_loss)
    rprint.color("white").style('bold').show(f"[Epoch {epoch + 1}/{epochs}] - Mean Loss: {avg_loss: .4f}")


print('\n')
plt.plot(losses)
