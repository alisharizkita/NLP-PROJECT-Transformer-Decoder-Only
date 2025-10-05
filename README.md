# NLP PROJECT-GPT Style Transformer (Decoder Only)

Name : Rizkita Alisha Ramadhani

NIM : 22/494942/TK/54347

Natural Language Processing Project

Syukron Abu Ishaq Alfarozi, S.T., Ph.D.

## 🚀 Transformer from Scratch

This project is an implementation of a **decoder-only Transformer (GPT-style)** built entirely from scratch using **NumPy**, without relying on any deep learning libraries such as TensorFlow or PyTorch. Its primary goal is to gain a deep understanding of the transformer architecture, the principles of *self-attention*, and the process of *autoregressive text generation*.
All core components of the model — including embedding, positional encoding, multi-head attention, feed-forward network, and causal masking — are implemented manually and validated through a series of unit tests, integration tests, and end-to-end pipeline evaluations.

## 🧠 Model Architecture
This model follows the basic GPT-style design with the following structure:

Input Layer

- Token Embedding: Maps token IDs into fixed-dimensional vector representations.

- Sinusoidal Positional Encoding: Injects positional information into the embeddings.

Transformer Blocks (2 Layers)

- Multi-Head Self-Attention (4 heads): Computes attention across tokens to capture contextual relationships.

- Feed-Forward Network: Two linear layers with a GELU activation function to transform the representations.

- Residual Connection + Layer Normalization: Improves training stability and accelerates convergence.

Output Layer

- Linear Projection: Projects the hidden states into the vocabulary space.

- Softmax: Produces the probability distribution for the next token prediction.

## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/username/transformer-from-scratch.git
cd transformer-from-scratch
```

### 2. Install the Dependencies

Make sure that Python ≥ 3.8 is installed, then intall all the dependencies required by :


```bash
pip install -r requirements.txt
```

### 3. Run Transformer

```bash
python transformer.py
```