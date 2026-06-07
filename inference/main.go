package main

import (
	"fmt"
	"log"
	"mini-gpt/internal/gguf"
	"mini-gpt/internal/tensor"
	"mini-gpt/internal/tokenizer"
	"os"
)

func main() {
	// 1. Setup Tokenizer (Must match training!)
	content, _ := os.ReadFile("data/onegin.txt")
	bpe := &tokenizer.BPE{}
	bpe.Train(string(content), 1000)

	// 2. Load Model
	f, err := os.Open("onegin.gguf")
	if err != nil {
		log.Fatalf("Failed to read GGUF file")
	}
	defer f.Close()
	gr := gguf.NewReader(f)
	gr.ReadAll()

	embd, _ := gr.LoadTensor("token_embd.weight")
	w1, _ := gr.LoadTensor("blk.0.attn_q.weight") // w1
	w2, _ := gr.LoadTensor("blk.0.attn_v.weight")
	wOut, _ := gr.LoadTensor("output.weight")

	// 3. Inference Parameters
	prompt := "Евгений"
	tokens := bpe.Encode(prompt)
	maxNewTokens := 20
	dModel := 64

	fmt.Printf("Prompt: %s\nGenerated: %s", prompt, prompt)

	for i := 0; i < maxNewTokens; i++ {
		// --- FORWARD PASS ---
		// We only care about the last token's prediction
		contextLen := len(tokens)
		x := tensor.NewTensor(1, contextLen, dModel)

		// Embedding Lookup
		for t := 0; t < contextLen; t++ {
			tokenID := tokens[t]
			for d := 0; d < dModel; d++ {
				x.Set(0, t, d, embd.At(0, tokenID, d))
			}
		}

		attn_out, _, _, _ := tensor.Attention(x, x, x)
		x_attn, _ := tensor.Add(x, attn_out)

		// Simplified Forward (Match your training architecture!)
		// For now, we'll just do: x -> w1 -> ReLU -> wOut
		h, _ := tensor.MatMul(x_attn, w1)
		tensor.ReLU(h)
		ffn_out, _ := tensor.MatMul(h, w2)
		x_final, _ := tensor.Add(x_attn, ffn_out)

		logits, _ := tensor.MatMul(x_final, wOut)
		tensor.Softmax(logits)

		// --- SAMPLING ---
		// We take the last row of logits (prediction for the next token)
		nextID := 0
		maxProb := -1.0
		for c := 0; c < logits.Cols; c++ {
			p := logits.At(0, contextLen-1, c)
			if p > maxProb {
				maxProb = p
				nextID = c
			}
		}

		// --- APPEND & PRINT ---
		tokens = append(tokens, nextID)
		fmt.Print(bpe.Decode([]int{nextID}))

		// Keep context length manageable
		if len(tokens) > 32 {
			tokens = tokens[1:]
		}
	}
	fmt.Println()
}
