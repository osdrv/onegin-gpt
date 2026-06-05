package main

import (
	"fmt"
	"mini-gpt/internal/tensor"
	"mini-gpt/internal/tokenizer"
)

func main() {
	sample := "hello"
	tk := tokenizer.NewTokenizer(sample)
	tokens := tk.Encode(sample)

	dModel := 8
	vocabSize := tk.VocabSize()
	lr := 0.1

	embeddings := tensor.NewTensor(vocabSize, dModel)
	embeddings.Randomize()

	wOut := tensor.NewTensor(dModel, vocabSize)
	wOut.Randomize()

	fmt.Println("Starting training...")

	for epoch := range 100 {
		totalLoss := 0.0
		for i := range len(tokens) - 1 {
			inputID := tokens[i]
			targetID := tokens[i+1]

			// Forward pass

			// Lookup
			x := tensor.NewTensor(1, dModel)
			for j := range dModel {
				x.Data[j] = embeddings.At(inputID, j)
			}

			// output
			logits, _ := tensor.MatMul(x, wOut)

			// probabilities
			tensor.Softmax(logits)

			// loss
			loss, _ := tensor.CrossEntropyLoss(logits, []int{targetID})
			totalLoss += loss

			// Backward pass

			// clear gradients
			embeddings.ZeroGrad()
			wOut.ZeroGrad()
			x.ZeroGrad()

			// backdrop through output layer
			tensor.MatMulBackward(x, wOut, logits)

			// backdrop to embeddings
			for j := range dModel {
				idx := inputID*dModel + j
				embeddings.Grad[idx] += x.Grad[j]
			}

			embeddings.Step(lr)
			wOut.Step(lr)
		}

		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(tokens)-1))
		}
	}
}
