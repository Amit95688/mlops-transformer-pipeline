
---
# Transformer From Scratch 🧠⚡

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Educational-success)

A **from-scratch PyTorch implementation of the Transformer architecture**, based on the paper  
📄 **“Attention Is All You Need” – Vaswani et al. (2017)**.

This repository focuses on **clarity, modularity, and learning**, avoiding high-level abstractions so you can understand *how Transformers actually work under the hood*.

---

## 🔍 What’s Inside?

✔ Complete **Encoder–Decoder Transformer**  
✔ **Multi-Head Self Attention** implemented manually  
✔ **Scaled Dot-Product Attention**  
✔ **Sinusoidal Positional Encoding**  
✔ Residual Connections + Layer Normalization  
✔ Clean, modular PyTorch code  
✔ Easy to extend for experiments

---

## 🏗️ Transformer Architecture

Input Embeddings ↓ Positional Encoding ↓ ┌──────────────────────────┐ │      Encoder Stack       │ │ ┌────────────────────┐  │ │ │ Multi-Head Attention│ │ │ │ Feed Forward        │ │ │ └────────────────────┘  │ └──────────────────────────┘ ↓ ┌──────────────────────────┐ │      Decoder Stack       │ │ ┌────────────────────┐  │ │ │ Masked Attention    │ │ │ │ Cross Attention     │ │ │ │ Feed Forward        │ │ │ └────────────────────┘  │ └──────────────────────────┘ ↓ Linear + Softmax

---

## 📦 Requirements

- Python **3.7+**
- PyTorch **1.9+**

Install PyTorch (example):

```bash
pip install torch


---

⚙️ Installation

git clone https://github.com/Amit95688/Transformer-From-Scratch.git
cd Transformer-From-Scratch


---
```
🚀 Quick Start
```
import torch
from model import build_transformer

model = build_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    src_seq_len=100,
    tgt_seq_len=100,
    d_model=512,
    d_ff=2048,
    num_heads=8,
    num_layers=6,
    dropout=0.1
)

src = torch.randint(0, 10000, (32, 20))
tgt = torch.randint(0, 10000, (32, 20))

enc_out = model.encode(src)
dec_out = model.decode(tgt, enc_out)
output = model.project(dec_out)

print(output.shape)  # (32, 20, 10000)

```
---

🧩 Core Components

🔹 Multi-Head Attention
```
from model import MultiHeadAttention

mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
```
🔹 Feed Forward Network
```
from model import FeedForward

ff = FeedForward(d_model=512, d_ff=2048, dropout=0.1)
```
🔹 Encoder Block
```
from model import EncoderBlock
```
encoder_block = EncoderBlock(d_model=512, self_attn=mha, feed_forward=ff, dropout=0.1)


---

🎭 Masking Utilities

Padding Mask → ignore padding tokens

Look-Ahead Mask → prevent future token leakage
```

def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1) == 0
```

---

🧪 Why This Repo?

This project is ideal if you want to:

Understand Transformers line-by-line

Prepare for research or interviews

Modify attention mechanisms

Build intuition before using nn.Transformer or HuggingFace


> This is not optimized for production — it’s optimized for learning.




---

📚 Reference

Vaswani et al., Attention Is All You Need, 2017

https://arxiv.org/abs/1706.03762



---

📜 License

MIT License — free to use, modify, and share.


---

👤 Author

Amit
🔗 https://github.com/Amit95688

If you found this helpful, consider ⭐ starring the repo!

---

