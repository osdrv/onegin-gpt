# Onegin-GPT 🚀

[![Go Version](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go)](https://go.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

<img src="https://github.com/osdrv/onegin-gpt/blob/main/web/nkshp.jpg" width="150" alt="A.S.P" />

A "bits-to-the-ground" implementation of a Generative Pre-trained Transformer (GPT) from absolute scratch in Go, specifically trained on Pushkin's *Eugene Onegin*. **Zero dependencies.** No PyTorch, no TensorFlow, no Gonum—just the standard library and pure mathematics.

```text
       .---.           .---.           .---.
      /     \         /     \         /     \
     |  BPE  | ----> | Attn  | ----> | GGUF  |
      \     /         \     /         \     /
       '---'           '---'           '---'
    Tokenizer       Transformer       Export
```

## ✨ Features

- **Pure Go Engine:** Every tensor operation (`MatMul`, `Softmax`, `LayerNorm`, `ReLU`) is implemented manually.
- **Manual Backpropagation:** No Autograd. All gradients are calculated by hand for every layer.
- **BPE Tokenization:** Custom Byte Pair Encoding (BPE) implementation with UTF-8 support (tested on Russian/Cyrillic).
- **Transformer Architecture:** Decoder-only block with Scaled Dot-Product Attention and Feed-Forward Networks.
- **GGUF v3 Support:** Export trained weights to `.gguf` files for use in industry-standard tools like LM Studio or llama.cpp.
- **SIMD-Friendly:** Optimized inner loops for performance on modern CPUs.

## 🏗️ Architecture

The model follows the classic GPT architecture scaled down for pedagogical purposes:

- **Embeddings:** Learned token embeddings.
- **Self-Attention:** Scaled dot-product attention mechanism.
- **FFN:** Multi-layer perceptron with ReLU activation.
- **Residual Connections:** Additive skip connections for stable gradient flow.
- **Output Layer:** Linear projection back to vocabulary space with Softmax.

## 🚀 Getting Started

### Prerequisites

- Go 1.22 or higher.

### 1. Training

The default training script trains on Pushkin's *Eugene Onegin* (`data/onegin.txt`).

```bash
go run main.go
```

This will:
1. Train the BPE tokenizer.
2. Initialize the Onegin-GPT weights.
3. Run the SGD training loop.
4. Export the final weights to `onegin.gguf`.

### 2. Inference

Once you have a `.gguf` file, you can run inference to generate text:

```bash
go run inference/main.go
```

## 🛠️ Project Structure

```text
.
├── data/               # Training corpora (e.g., onegin.txt)
├── internal/
│   ├── tensor/         # Core Math: MatMul, Gradients, ReLU, Attention, etc.
│   ├── tokenizer/      # BPE implementation
│   ├── gguf/           # GGUF v3 Read/Write logic
│   └── data/           # Dataset loading & Batching
├── main.go             # Training entry point
└── inference/          # Text generation script
```

## 📜 Why?

Most people treat LLMs as "magic boxes." This project is a pedagogical exercise to prove that a Transformer is just a series of matrix multiplications and clever calculus. By implementing it in a type-safe, compiled language like Go, we gain deep intuition about memory layout, floating-point precision, and the sheer elegance of the attention mechanism.

---

*“What I cannot create, I do not understand.” – Richard Feynman*
