# Transformer NMT (English → Hindi)

A PyTorch implementation of the *Attention Is All You Need* architecture for English–Hindi neural machine translation.  
Inspired by the original Transformer paper and influenced by the style of the GPT-2 GitHub repository.

---

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Regex
- Jupyter (for running notebooks)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Training Notes

- This model was trained on **20,000 English–Hindi sentence pairs**.
- For optimal performance, training on a larger dataset is recommended.

## References

- Vaswani et al., [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762), 2017  
- [Transformer (Wikipedia)](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))  
- [GPT-2 GitHub Repo](https://github.com/openai/gpt-2) (inspiration for repo structure and documentation style)
