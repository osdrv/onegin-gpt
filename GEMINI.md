# Mini-GPT in Go: Roadmap & Context

## Project Vision
Build a simplistic Large Language Model (LLM) from scratch in Go to understand the "bits-to-the-ground" internals, including attention mechanisms, model file structures (GGUF), and hardware acceleration on Apple Silicon (M2).

## Current Status: Phase 1 (The "Naked" GPT)
- **Tokenization:** Character-level (UTF-8 support for Eugene Onegin).
- **Architecture:** Transformer (Decoder-only).
- **Training:** Manual backpropagation implementation.
- **Hardware:** CPU (Initial) -> Metal/MLX (Later).

## Roadmap

### Phase 1: The "Naked" GPT (Pure Go, CPU)
- [ ] Character-level Tokenizer (UTF-8).
- [ ] Manual Tensor/Matrix operations (`MatMul`, `Softmax`, etc.).
- [ ] Scaled Dot-Product Attention implementation.
- [ ] Transformer Block (LayerNorm, Feed-Forward).
- [ ] Manual Backpropagation & Training Loop.

### Phase 2: Scaling & Data
- [ ] Byte Pair Encoding (BPE) Tokenizer.
- [ ] Sliding Window Data Loading.
- [ ] Performance optimization (Gonum/SIMD).

### Phase 3: Serialization & Interop
- [ ] GGUF v3 Writer implementation.
- [ ] Export weights to GGUF and verify in LM Studio.

### Phase 4: Hardware Acceleration (Mac M2)
- [ ] Metal / MLX Go bindings investigation.
- [ ] Offload heavy computations to GPU.
- [ ] Unified Memory management.

### Phase 5: Application
- [ ] OpenAI-style Chat API.
- [ ] Zed Editor integration.

## Design Principles
- **No Magic:** Minimize libraries; implement core math manually for learning.
- **Pedagogical:** Code snippets are provided for manual entry to ensure deep understanding.
- **UTF-8 Native:** Handle Cyrillic and other multi-byte characters from day one.
