package tensor

import "testing"

func TestAttention(t *testing.T) {
	// Simple identity-like test
	// Q: [1 0], K: [1 0; 0 1], V: [10 20; 30 40]
	q := NewTensor(1, 2)
	q.Data = []float64{1, 0}

	k := NewTensor(2, 2)
	k.Data = []float64{1, 0, 0, 1}

	v := NewTensor(2, 2)
	v.Data = []float64{1, 0, 0, 1}

	out, err := Attention(q, k, v)
	if err != nil {
		t.Fatalf("Attention failed: %v", err)
	}

	// Since Q matches the first row of K, the first row of V should dominate.
	// With Softmax and scaling, it won't be EXACTLY 10 and 20, but they should be much larger.
	if out.Data[0] < out.Data[1] {
		t.Errorf("Attention failed to weight V correctly: %v", out.Data)
	}
}
