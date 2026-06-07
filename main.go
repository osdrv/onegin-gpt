package main

import (
	"log"
	"mini-gpt/internal/data"
	"mini-gpt/internal/gguf"
	"mini-gpt/internal/tensor"
	"mini-gpt/internal/tokenizer"
	"os"
)

func main() {
	// 1. Load and Tokenize
	log.Println("Loading the dataset...")
	content, err := os.ReadFile("data/onegin.txt")
	if err != nil {
		log.Fatalf("Failed to read dataset file: %s", err)
	}
	text := string(content)

	log.Println("Training BPE (1000 merges)...")
	bpe := &tokenizer.BPE{}
	bpe.Train(text, 1000)

	tokens := bpe.Encode(text)
	dataset := data.NewDataset(tokens)

	// 2. Hyperparameters
	batchSize := 16
	contextLen := 32
	dModel := 64
	vocabSize := 256 + 1000
	lr := 1.0 // High learning rate for "Naked" SGD

	// 3. Initialize Weights
	embeddings := tensor.NewTensor(1, vocabSize, dModel)
	embeddings.Randomize()

	wOut := tensor.NewTensor(1, dModel, vocabSize)
	wOut.Randomize()

	w1 := tensor.NewTensor(1, dModel, dModel*4)
	w1.Randomize()
	w2 := tensor.NewTensor(1, dModel*4, dModel)
	w2.Randomize()

	log.Printf("Model initialized. Vocab: %d, Batch: %d, Context: %d", vocabSize, batchSize, contextLen)

	// 4. Training loop
	for step := 0; step < 5000; step++ {
		// --- 0. PREPARE ---
		embeddings.ZeroGrad()
		wOut.ZeroGrad()
		w1.ZeroGrad()
		w2.ZeroGrad()

		x_tokens, y_tokens := dataset.GetBatch(batchSize, contextLen)

		// --- 1. FORWARD PASS ---

		// Embedding Lookup
		x := tensor.NewTensor(batchSize, contextLen, dModel)
		for b := 0; b < batchSize; b++ {
			for t := 0; t < contextLen; t++ {
				tokenId := x_tokens[b][t]
				for d := 0; d < dModel; d++ {
					x.Set(b, t, d, embeddings.At(0, tokenId, d))
				}
			}
		}

		// Self-Attention
		attn_out, probs, scores, _ := tensor.Attention(x, x, x)
		x_attn, _ := tensor.Add(x, attn_out)

		// FFN
		h, _ := tensor.MatMul(x_attn, w1)
		tensor.ReLU(h)
		ffn_out, _ := tensor.MatMul(h, w2)

		// Residual Connection
		x_final, _ := tensor.Add(x_attn, ffn_out)

		// Output Layer
		logits, _ := tensor.MatMul(x_final, wOut)
		tensor.Softmax(logits)
		loss, _ := tensor.CrossEntropyLoss(logits, y_tokens)

		// --- 2. BACKWARD PASS ---

		// Output Layer
		tensor.MatMulBackward(x_final, wOut, logits)

		// FFN & Residual
		tensor.AddBackward(x_attn, ffn_out, x_final)
		tensor.MatMulBackward(h, w2, ffn_out)
		tensor.ReLUBackward(h, h)
		tensor.MatMulBackward(x_attn, w1, h)

		// Attention & Residual
		tensor.AddBackward(x, attn_out, x_attn)
		tensor.AttentionBackward(x, x, x, probs, scores, attn_out)

		// Backprop into Master Embedding Table
		for b := 0; b < batchSize; b++ {
			for t := 0; t < contextLen; t++ {
				tokenId := x_tokens[b][t]
				for d := 0; d < dModel; d++ {
					grad := x.GradAt(b, t, d)
					embeddings.AddGrad(0, tokenId, d, grad)
				}
			}
		}

		// --- 3. UPDATE ---
		embeddings.Step(lr)
		wOut.Step(lr)
		w1.Step(lr)
		w2.Step(lr)

		if step%50 == 0 {
			log.Printf("Step %d, Loss: %.4f", step, loss)
		}
	}

	log.Println("Saving model to onegin.gguf...")
	f, err := os.Create("onegin.gguf")
	if err != nil {
		log.Fatalf("Failed to save model output: %s", err)
	}
	defer f.Close()

	gw := gguf.NewWriter(f)

	gw.SetMetadta("general.architecture", "mini-gpt")
	gw.SetMetadta("general.name", "Onegin-GPT")
	gw.SetMetadta("mini-gpt.context_length", "32")
	gw.SetMetadta("mini-gpt.embedding_length", "64")

	gw.AddTensor("token_embd.weight", []uint64{uint64(dModel), uint64(vocabSize)}, embeddings.Data)
	gw.AddTensor("blk.0.attn_q.weight", []uint64{uint64(dModel), uint64(dModel)}, w1.Data)
	gw.AddTensor("blk.0.attn_v.weight", []uint64{uint64(dModel), uint64(dModel)}, w2.Data)
	gw.AddTensor("output.weight", []uint64{uint64(vocabSize), uint64(dModel)}, wOut.Data)

	if err := gw.WriteHeaders(); err != nil {
		log.Fatalf("GGUF WriteHeaders failed: %s", err)
	}

	gw.WritePadding()

	log.Println("Writing tensor data...")
	gw.WriteData(embeddings.Data)
	gw.WriteData(w1.Data)
	gw.WriteData(wOut.Data)

	log.Println("Successfully saved Onegin-GPT model")
}
