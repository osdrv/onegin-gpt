package tensor

import (
	"fmt"
	"math"
	"math/rand"
)

type Tensor struct {
	Data  []float64
	Grad  []float64
	Batch int
	Rows  int
	Cols  int
}

func NewTensor(batch, rows, cols int) *Tensor {
	return &Tensor{
		Data:  make([]float64, batch*rows*cols),
		Grad:  make([]float64, batch*rows*cols),
		Batch: batch,
		Rows:  rows,
		Cols:  cols,
	}
}

func (t *Tensor) At(b, r, c int) float64 {
	return t.Data[b*t.Rows*t.Cols+r*t.Cols+c]
}

func (t *Tensor) Set(b, r, c int, val float64) {
	t.Data[b*t.Rows*t.Cols+r*t.Cols+c] = val
}

func MatMul(a, b *Tensor) (*Tensor, error) {
	if a.Cols != b.Rows {
		return nil, fmt.Errorf("incompatible dimensions: %dx%d Vs %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	out := NewTensor(a.Batch, a.Rows, b.Cols)

	aStride := a.Rows * a.Cols
	bStride := b.Rows * b.Cols
	outStride := out.Rows * out.Cols

	for bix := 0; bix < a.Batch; bix++ {
		aOff := bix * aStride
		bOff := 0
		if b.Batch > 1 {
			bOff = bix * bStride
		}
		outOff := bix * outStride

		for i := 0; i < a.Rows; i++ {
			aRowStart := aOff + i*a.Cols
			outRowStart := outOff + i*out.Cols

			for k := 0; k < a.Cols; k++ {
				av := a.Data[aRowStart+k]
				if av == 0 {
					continue
				}
				bRowStart := bOff + k*b.Cols

				// SIMD reading
				for j := 0; j < b.Cols; j++ {
					out.Data[outRowStart+j] += av * b.Data[bRowStart+j]
				}
			}
		}
	}

	return out, nil
}

// In a Transformer, we use this for residual connections
func Add(a, b *Tensor) (*Tensor, error) {
	if len(a.Data) != len(b.Data) {
		return nil, fmt.Errorf("size mismatch: %d Vs %d", len(a.Data), len(b.Data))
	}

	out := NewTensor(a.Batch, a.Rows, a.Cols)

	aData := a.Data
	bData := b.Data
	outData := out.Data

	for i := 0; i < len(aData); i++ {
		outData[i] = aData[i] + bData[i]
	}

	return out, nil
}

// We need this because attention calculates QK^T
func Transpose(t *Tensor) *Tensor {
	out := NewTensor(t.Batch, t.Cols, t.Rows)
	for bix := range t.Batch {
		for r := range t.Rows {
			for c := range t.Cols {
				out.Set(bix, c, r, t.At(bix, r, c))
			}
		}
	}
	return out
}

// Softmax turns a list of numbers into probabilities that sum to 1.
// In a Transformer, it determines which words the model should "attend" to.
func Softmax(t *Tensor) {
	for bix := range t.Batch {
		for r := range t.Rows {
			max := t.At(bix, r, 0)
			for c := 1; c < t.Cols; c++ {
				if t.At(bix, r, c) > max {
					max = t.At(bix, r, c)
				}
			}
			sum := 0.0
			for c := range t.Cols {
				val := math.Exp(t.At(bix, r, c) - max)
				t.Set(bix, r, c, val)
				sum += val
			}

			for c := range t.Cols {
				t.Set(bix, r, c, t.At(bix, r, c)/sum)
			}
		}
	}
}

func SoftmaxBackward(probs, out *Tensor) {
	for bix := 0; bix < probs.Batch; bix++ {
		for r := 0; r < probs.Rows; r++ {
			dot := 0.0
			for c := 0; c < probs.Cols; c++ {
				dot += probs.At(bix, r, c) * out.GradAt(bix, r, c)
			}
			for c := 0; c < probs.Cols; c++ {
				p := probs.At(bix, r, c)
				g := out.GradAt(bix, r, c)
				probs.AddGrad(bix, r, c, p*(g-dot))
			}
		}
	}
}

// LayerNorm works by making the mean of a row 0 and its variance 1.
// To do this, we need to calculate the average and deviation for each row.
func (t *Tensor) RowMeanVar(b, r int) (mean, variance float64) {
	sum := 0.0
	for c := 0; c < t.Cols; c++ {
		sum += t.At(b, r, c)
	}
	mean = sum / float64(t.Cols)

	sqDiffSum := 0.0
	for c := 0; c < t.Cols; c++ {
		diff := t.At(b, r, c) - mean
		sqDiffSum += diff * diff
	}
	variance = sqDiffSum / float64(t.Cols)
	return mean, variance
}

func LayerNorm(t *Tensor) {
	eps := 1e-5
	for bix := 0; bix < t.Batch; bix++ {
		for r := 0; r < t.Rows; r++ {
			mean, variance := t.RowMeanVar(bix, r)
			std := math.Sqrt(variance + eps)
			for c := 0; c < t.Cols; c++ {
				val := (t.At(bix, r, c) - mean) / std
				t.Set(bix, r, c, val)
			}
		}
	}
}

func LayerNormBackward(input, out *Tensor) {
	eps := 1e-5
	for bix := 0; bix < input.Batch; bix++ {
		for r := 0; r < input.Rows; r++ {
			mean, variance := input.RowMeanVar(bix, r)
			std := math.Sqrt(variance + eps)

			var gradSum, gradDot float64
			for c := 0; c < input.Cols; c++ {
				g := out.GradAt(bix, r, c)
				gradSum += g
				gradDot += g * (input.At(bix, r, c) - mean)
			}

			for c := 0; c < input.Cols; c++ {
				g := out.GradAt(bix, r, c)
				term1 := float64(input.Cols) * g
				term2 := gradSum
				term3 := (input.At(bix, r, c) - mean) * gradDot / (variance + eps)

				input.AddGrad(bix, r, c, (term1-term2-term3)/(float64(input.Cols)*std))
			}
		}
	}
}

// If a number is negative, make it 0. This is the "brain" that allows
// the model to learn non-linear patterns.
func ReLU(t *Tensor) {
	data := t.Data // cache locally
	for i := 0; i < len(data); i++ {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

// The rule for ReLU is: if the input was positive, the gradient passes
// through unchanged. If the input was negative or zero, the gradient
// is blocked (set to 0).
func ReLUBackward(input, out *Tensor) {
	inData := input.Data
	inGrad := input.Grad
	outGrad := out.Grad
	for i := 0; i < len(inData); i++ {
		if inData[i] > 0 {
			inGrad[i] += outGrad[i]
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
//   - grad_B = Σ (A^T × grad_C)
func MatMulBackward(a, b, out *Tensor) {
	for bix := 0; bix < out.Batch; bix++ {
		// 1. grad_A = grad_out * B^T
		for i := 0; i < a.Rows; i++ {
			for k := 0; k < out.Cols; k++ {
				gOk := out.GradAt(bix, i, k)
				for j := 0; j < a.Cols; j++ {
					a.AddGrad(bix, i, j, gOk*b.At(0, j, k))
				}
			}
		}
		// 2. grad_B = Σ (A^T * grad_out)
		for i := 0; i < b.Rows; i++ {
			for k := 0; k < a.Rows; k++ {
				av := a.At(bix, k, i)
				if av == 0 {
					continue
				}
				for j := 0; j < b.Cols; j++ {
					b.AddGrad(0, i, j, av*out.GradAt(bix, k, j))
				}
			}
		}
	}
}

func (t *Tensor) ZeroGrad() {
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

func (t *Tensor) GradAt(b, r, c int) float64 {
	return t.Grad[b*t.Rows*t.Cols+r*t.Cols+c]
}

func (t *Tensor) SetGrad(b, r, c int, val float64) {
	t.Grad[b*t.Rows*t.Cols+r*t.Cols+c] = val
}

func (t *Tensor) AddGrad(b, r, c int, val float64) {
	t.Grad[b*t.Rows*t.Cols+r*t.Cols+c] += val
}

// CrossEntropyLoss calculates the loss and sets gradients on 'probs'
// 'probs' is the output of Softmax, 'targets' is a 2D slice [batch][seq]
func CrossEntropyLoss(probs *Tensor, targets [][]int) (float64, error) {
	loss := 0.0
	probs.ZeroGrad()

	totalExamples := float64(probs.Batch * probs.Rows)

	for bix := 0; bix < probs.Batch; bix++ {
		for r := 0; r < probs.Rows; r++ {
			targetID := targets[bix][r]
			prob := probs.At(bix, r, targetID)
			loss -= math.Log(prob + 1e-10)

			for c := 0; c < probs.Cols; c++ {
				p := probs.At(bix, r, c)
				if c == targetID {
					probs.SetGrad(bix, r, c, (p-1.0)/totalExamples)
				} else {
					probs.SetGrad(bix, r, c, p/totalExamples)
				}
			}
		}
	}
	return loss / totalExamples, nil
}

func (t *Tensor) Step(lr float64) {
	data := t.Data
	grad := t.Grad
	for i := 0; i < len(data); i++ {
		data[i] -= lr * grad[i]
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
