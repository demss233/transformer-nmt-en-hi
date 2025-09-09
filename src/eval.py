import torch
import torch.nn as nn

def translate_sentence(sentence, model, source_vocab, target_vocab, max_len = 50, device, pad_sequence, tokenize, max_sequence_length = 40):
    model.eval()
    idx2word = {i: w for w, i in target_vocab.items()}
    source_tokens = tokenize(sentence, source_vocab)
    padded_source = pad_sequence(source_tokens, max_sequence_length)
    source_tensor = torch.from_numpy(padded_source).to(device).unsqueeze(0)
    sos_idx = target_vocab["<sos>"]
    eos_idx = target_vocab["<eos>"]
    target_indices = [sos_idx]

    with torch.no_grad():
        for _ in range(max_len):
            target_tensor = torch.tensor(target_indices).to(device).unsqueeze(0)
            outputs = model(source_tensor, target_tensor)
            next_token = outputs[:, -1, :].argmax(dim = -1).item()
            target_indices.append(next_token)

            if next_token == eos_idx:
                break

    return " ".join(idx2word[i] for i in target_indices[1: -1])