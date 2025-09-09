import torch
import torch.nn as nn
import math
import numpy as np

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.lookup = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first = True)

    def forward(self, x):
        out = self.lookup(x)
        out, hidden = self.gru(out)
        return out, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias = False)

    def forward(self, hidden, output):
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.repeat(1, output.shape[1], 1)
        energy = torch.tanh(self.fc1(torch.cat((hidden, output), dim = 2)))
        return torch.softmax(self.fc2(energy).squeeze(2), dim = 1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.lookup = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(
            embedding_dim + hidden_dim,
            hidden_dim,
            batch_first = True
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        out = self.lookup(x)
        attention_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)

        context = torch.bmm(attention_weights, encoder_outputs)

        if out.dim() == 2:
            out = out.unsqueeze(1)

        if context.dim() == 2:
            context = context.unsqueeze(1)

        _ = torch.cat((out, context), dim = 2)
        out, hidden = self.gru(_, hidden)
        out = out.squeeze(1)
        context = context.squeeze(1)

        out = self.fc(torch.cat((out, context), dim = 1))
        return out, hidden, attention_weights.squeeze(1)

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding from 
    'Attention Is All You Need' (Vaswani et al., 2017).

    Since the Transformer has no recurrence or convolution,
    it needs a way to represent token positions in a sequence.
    This encoding injects position information into embeddings
    using sine and cosine functions at different frequencies.

    The resulting [seq_len, d_model] tensor is deterministic
    and bounded in [-1, 1]. These values are usually stored
    as a non-trainable buffer for efficiency.
    """
    def __init__(self, d, dropout, max_length = 200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_length, d)
        pos = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2Seq(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 embedding_dim,
                 n_heads,
                 source_vocab_size,
                 target_vocab_size,
                 dim_feedforward = 512,
                 dropout = 0.1,
                 pad_idx = 0
                ):
        super(Seq2Seq, self).__init__()
        self.pad_idx = pad_idx

        self.transformer = nn.Transformer(
            d_model = embedding_dim,
            nhead = n_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True
        )

        self.generator = nn.Linear(embedding_dim, target_vocab_size)
        self.source_token_embedding = TokenEmbedding(source_vocab_size, embedding_dim)
        self.target_token_embedding = TokenEmbedding(target_vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)

    def make_pad_mask(self, seq):
        return (seq == self.pad_idx)

    def make_causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device = device), diagonal = 1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, src, trg):
        src_emb = self.positional_encoding(self.source_token_embedding(src))
        trg_emb = self.positional_encoding(self.target_token_embedding(trg))

        src_pad_mask = self.make_pad_mask(src)
        trg_pad_mask = self.make_pad_mask(trg)
        tgt_mask = self.make_causal_mask(trg.size(1), trg.device)

        outs = self.transformer(
            src_emb,
            trg_emb,
            tgt_mask = tgt_mask,
            src_key_padding_mask = src_pad_mask,
            tgt_key_padding_mask = trg_pad_mask,
            memory_key_padding_mask = src_pad_mask
        )
        return self.generator(outs)