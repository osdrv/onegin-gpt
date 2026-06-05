package tensor

import (
	"fmt"
	"math"
	"math/rand"
)

type Tensor struct {
	Data []float64
	Grad []float64
	Rows int
	Cols int
}

func NewTensor(rows, cols int) *Tensor {
	return &Tensor{
		Data: make([]float64, rows*cols),
		Grad: make([]float64, rows*cols),
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

func SoftmaxBackward(probs, out *Tensor) {
	for r := range probs.Rows {
		dot := 0.0
		for c := range probs.Cols {
			dot += probs.At(r, c) * out.Grad[r*probs.Cols+c]
		}
		for c := range probs.Cols {
			p := probs.At(r, c)
			g := out.Grad[r*probs.Cols+c]
			probs.Grad[r*probs.Cols+c] += p * (g - dot)
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

func LayerNormBackward(input, out *Tensor) {
	eps := 1e-5
	for r := range input.Rows {
		mean, variance := input.RowMeanVar(r)
		std := math.Sqrt(variance + eps)

		var gradSum, gradDot float64
		for c := range input.Cols {
			g := out.Grad[r*input.Cols+c]
			gradSum += g
			gradDot += g * (input.At(r, c) - mean)
		}

		for c := range input.Cols {
			g := out.Grad[r*input.Cols+c]
			term1 := float64(input.Cols) * g
			term2 := gradSum
			term3 := (input.At(r, c) - mean) * gradDot / (variance + eps)

			input.Grad[r*input.Cols+c] += (term1 - term2 - term3) / (float64(input.Cols) * std)
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

// The rule for ReLU is: if the input was positive, the gradient passes
// through unchanged. If the input was negative or zero, the gradient
// is blocked (set to 0).
func ReLUBackward(input, out *Tensor) {
	for i := range input.Data {
		if input.Data[i] > 0 {
			input.Grad[i] += out.Grad[i]
		}
	}
}

// Backpropagation for Addition
// For C = A + B, the gradient flows back equally: grad_A = grad_C and grad_B = grad_C.
func AddBackward(a, b, out *Tensor) {
	for i := range out.Grad {
		a.Grad[i] += out.Grad[i]
		b.Grad[i] += out.Grad[i]
	}
}

// For C = A × B:
//   - grad_A = grad_C × B^T
//   - grad_B = A^T × grad_C
func MatMulBackward(a, b, out *Tensor) {
	for i := range a.Rows {
		for j := range a.Cols {
			sum := 0.0
			for k := range out.Cols {
				sum += out.Grad[i*out.Cols+k] * b.Data[j*b.Cols+k]
			}
			a.Grad[i*a.Cols+j] += sum
		}
	}

	for i := range b.Rows {
		for j := range b.Cols {
			sum := 0.0
			for k := range out.Rows {
				sum += a.Data[k*a.Cols+i] * out.Grad[k*out.Cols+j]
			}
			b.Grad[i*b.Cols+j] += sum
		}
	}
}

func (t *Tensor) ZeroGrad() {
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

// CrossEntropyLoss calculates the loss and sets gradients on 'probs'
// 'probs' is the output of Softmax, 'targets' is a slice of correct token IDs
func CrossEntropyLoss(probs *Tensor, targets []int) (float64, error) {
	loss := 0.0
	probs.ZeroGrad()

	for r := range probs.Rows {
		targetID := targets[r]
		// Loss = -log(probability of the correct class)
		prob := probs.At(r, targetID)
		// We add a tiny epsilon to avoid log(0)
		loss -= math.Log(prob + 1e-10)

		// Gradient of (Softmax + CrossEntropy) is simply: prob - 1
		// (for the correct class)
		// For all other classes, it's just: prob - 0
		for c := range probs.Cols {
			p := probs.At(r, c)
			if c == targetID {
				probs.Grad[r*probs.Cols+c] = (p - 1.0) / float64(probs.Rows)
			} else {
				probs.Grad[r*probs.Cols+c] = p / float64(probs.Rows)
			}
		}
	}
	return loss / float64(probs.Rows), nil
}

func (t *Tensor) Step(lr float64) {
	for i := range t.Data {
		t.Data[i] -= lr * t.Grad[i]
	}
}

// Xavier Initialization
//
//	The formula is to pick numbers from a uniform distribution between [-L, L], where:
//	L = √((6)/(fan_in + fan_out))
//	 * fan_in: Number of input neurons (rows).
//	 * fan_out: Number of output neurons (cols).
func (t *Tensor) Randomize() {
	//rand.Seed(time.Now().UnixNano())

	fanIn := float64(t.Rows)
	fanOut := float64(t.Cols)
	limit := math.Sqrt(6.0 / (fanIn + fanOut))

	for i := range t.Data {
		t.Data[i] = (rand.Float64()*2 - 1) * limit
	}
}
