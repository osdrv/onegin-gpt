package tensor

import "math"

// Input: Three tensors: Q (Query), K (Key), and V (Value).
// Step 1 (QK^T): Multiply Q by the transpose of K. This calculates the "raw" affinity
// between words.
// Step 2 (Scaling): Multiply every value in the resulting matrix by 1 / √(dₖ) (where dₖ is
// the dimension of the keys). This prevents gradients from getting too small.
// Step 3 (Softmax): Apply Softmax to the scaled matrix to get probabilities.
// Step 4 (Multiply by V): Multiply the probabilities by V to get the final output.
func Attention(q, k, v *Tensor) (*Tensor, error) {
	kt := Transpose(k)
	scores, err := MatMul(q, kt)
	if err != nil {
		return nil, err
	}

	dk := float64(k.Cols)
	scale := 1.0 / math.Sqrt(dk)
	for i := range scores.Data {
		scores.Data[i] *= scale
	}

	Softmax(scores)

	return MatMul(scores, v)
}
