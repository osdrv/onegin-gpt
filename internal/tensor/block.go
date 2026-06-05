package tensor

// The Feed-Forward Network (FFN) consists of:
//  1. Linear Layer 1: Expands the dimension (usually 4x).
//  2. ReLU: Non-linearity.
//  3. Linear Layer 2: Contracts back to the original dimension.
func FFN(input, w1, w2 *Tensor) (*Tensor, error) {
	// First linear layer
	h, err := MatMul(input, w1)
	if err != nil {
		return nil, err
	}
	// 2. Activation
	ReLU(h)
	// 3. Second linear layer
	return MatMul(h, w2)
}

func TransformerBlock(x, wq, wk, wv, w1, w2 *Tensor) (*Tensor, error) {
	// 1. Self-attention sub-layer
	x_norm := &Tensor{
		Data: append([]float64(nil), x.Data...), Rows: x.Rows, Cols: x.Cols,
	}
	LayerNorm(x_norm)

	// attention_out = Attention(Q, K, V)
	attn, err := Attention(x_norm, x_norm, x_norm) // the simplest self-attention
	if err != nil {
		return nil, err
	}

	x, _ = Add(x, attn)
	x_norm_2 := &Tensor{
		Data: append([]float64(nil), x.Data...), Rows: x.Rows, Cols: x.Cols,
	}
	LayerNorm(x_norm_2)

	ffn_out, err := FFN(x_norm_2, w1, w2)
	if err != nil {
		return nil, err
	}

	return Add(x, ffn_out)
}
