from .utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def initialize(source_vocab, target_vocab, sentences, device):
    class Configure(torch.utils.data.Dataset):
        def __init__(self, data, max_len, source_vocab, target_vocab):
            self.data = data
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            target = self.data.iloc[index]["hindi"]
            source = self.data.iloc[index]["english"]
            target_tokens = tokenize(target, target_vocab)
            source_tokens = tokenize(source, source_vocab)
            padded_target = pad_sequence(target_tokens, self.max_len)
            padded_source = pad_sequence(source_tokens, self.max_len)
            target_tensor = torch.from_numpy(padded_target).long().to(device)
            source_tensor = torch.from_numpy(padded_source).long().to(device)
            return source_tensor, target_tensor

    batch_size = 20
    num_workers = 0
    max_sequence_length = 40

    dataset = Configure(
        data = sentences,
        max_len = max_sequence_length,
        source_vocab = source_vocab,
        target_vocab = target_vocab
    )

    dataloader = DataLoader(dataset, batch_size, num_workers = num_workers, shuffle = True)
    return dataloader