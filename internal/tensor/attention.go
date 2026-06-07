package tensor

import "math"

// Attention performs the Scaled Dot-Product Attention: Attention(Q, K, V) = Softmax(QK^T / sqrt(dk))V
func Attention(q, k, v *Tensor) (*Tensor, *Tensor, *Tensor, error) {
	// 1. Q * K^T
	kt := Transpose(k)
	scores, err := MatMul(q, kt)
	if err != nil {
		return nil, nil, nil, err
	}

	// 2. Scale by sqrt(dk)
	dk := float64(k.Cols)
	scale := 1.0 / math.Sqrt(dk)
	for i := range scores.Data {
		scores.Data[i] *= scale
	}

	// 3. Softmax
	// We create a copy for probs to keep scores intact for backward pass
	probs := &Tensor{
		Data:  append([]float64(nil), scores.Data...),
		Grad:  make([]float64, len(scores.Data)),
		Batch: scores.Batch,
		Rows:  scores.Rows,
		Cols:  scores.Cols,
	}
	Softmax(probs)

	// 4. Multiply by V
	out, err := MatMul(probs, v)
	return out, probs, scores, err
}

// AttentionBackward propagates gradients back through the attention mechanism.
func AttentionBackward(q, k, v, probs, scores, out *Tensor) {
	// 1. Backward through out = MatMul(probs, v)
	// Updates probs.Grad and v.Grad
	MatMulBackward(probs, v, out)

	// 2. Backward through probs = Softmax(scores)
	// SoftmaxBackward updates probs.Grad in-place to become the gradient w.r.t scores
	SoftmaxBackward(probs, probs)

	// 3. Backward through scaling: scores = raw_scores * scale
	dk := float64(k.Cols)
	scale := 1.0 / math.Sqrt(dk)
	// We transfer gradients from the softmax output back to the raw scores
	for i := range scores.Grad {
		scores.Grad[i] = probs.Grad[i] * scale
	}

	// 4. Backward through raw_scores = MatMul(q, k^T)
	kt := Transpose(k)
	// Note: kt.Grad is initialized to zero by Transpose via NewTensor
	MatMulBackward(q, kt, scores)

	// 5. Accumulate kt.Grad back into k.Grad (Reverse Transpose)
	for b := 0; b < k.Batch; b++ {
		for r := 0; r < k.Rows; r++ {
			for c := 0; c < k.Cols; c++ {
				k.AddGrad(b, r, c, kt.GradAt(b, c, r))
			}
		}
	}
}
