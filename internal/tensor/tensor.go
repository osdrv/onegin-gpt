package tensor

import (
	"fmt"
	"math"
)

type Tensor struct {
	Data []float64
	Rows int
	Cols int
}

func NewTensor(rows, cols int) *Tensor {
	return &Tensor{
		Data: make([]float64, rows*cols),
		Rows: rows,
		Cols: cols,
	}
}

func (t *Tensor) At(r, c int) float64 {
	return t.Data[r*t.Cols+c]
}

func (t *Tensor) Set(r, c int, val float64) {
	t.Data[r*t.Cols+c] = val
}

func MatMul(a, b *Tensor) (*Tensor, error) {
	if a.Cols != b.Rows {
		return nil, fmt.Errorf("incompatible dimensions: %dx%d Vs %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	out := NewTensor(a.Rows, b.Cols)
	for i := range a.Rows {
		for j := range b.Cols {
			sum := 0.0
			for k := range a.Cols {
				sum += a.At(i, k) * b.At(k, j)
			}
			out.Set(i, j, sum)
		}
	}
	return out, nil
}

// In a Transformer, we use this for residual connections
func Add(a, b *Tensor) (*Tensor, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, fmt.Errorf("dimensions mismatch: %dx%d Vs %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	out := NewTensor(a.Rows, a.Cols)
	for i := range a.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out, nil
}

// We need this because attention calculates QK^T
func Transpose(t *Tensor) *Tensor {
	out := NewTensor(t.Cols, t.Rows)
	for r := range t.Rows {
		for c := range t.Cols {
			out.Set(c, r, t.At(r, c))
		}
	}
	return out
}

// Softmax turns a list of numbers into probabilities that sum to 1.
// In a Transformer, it determines which words the model should "attend" to.
func Softmax(t *Tensor) {
	for r := range t.Rows {
		max := t.At(r, 0)
		for c := 1; c < t.Cols; c++ {
			if t.At(r, c) > max {
				max = t.At(r, c)
			}
		}
		sum := 0.0
		for c := range t.Cols {
			val := math.Exp(t.At(r, c) - max)
			t.Set(r, c, val)
			sum += val
		}

		for c := range t.Cols {
			t.Set(r, c, t.At(r, c)/sum)
		}
	}
}

// LayerNorm works by making the mean of a row 0 and its variance 1.
// To do this, we need to calculate the average and deviation for each row.
func (t *Tensor) RowMeanVar(r int) (mean, variance float64) {
	sum := 0.0
	for c := range t.Cols {
		sum += t.At(r, c)
	}
	mean = sum / float64(t.Cols)

	sqDiffSum := 0.0
	for c := range t.Cols {
		diff := t.At(r, c) - mean
		sqDiffSum += diff * diff
	}
	variance = sqDiffSum / float64(t.Cols)
	return mean, variance
}

func LayerNorm(t *Tensor) {
	eps := 1e-5
	for r := range t.Rows {
		mean, variance := t.RowMeanVar(r)
		std := math.Sqrt(variance + eps)
		for c := range t.Cols {
			val := (t.At(r, c) - mean) / std
			t.Set(r, c, val)
		}
	}
}

// If a number is negative, make it 0. This is the "brain" that allows
// the model to learn non-linear patterns.
func ReLU(t *Tensor) {
	for i := range t.Data {
		if t.Data[i] < 0 {
			t.Data[i] = 0
		}
	}
}
