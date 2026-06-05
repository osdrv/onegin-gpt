package tensor

import (
	"math"
	"testing"
)

func TestMatMul(t *testing.T) {
	// A = [1 2; 3 4]
	a := NewTensor(2, 2)
	a.Data = []float64{1, 2, 3, 4}

	// B = [5 6; 7 8]
	b := NewTensor(2, 2)
	b.Data = []float64{5, 6, 7, 8}

	// Expected C = A * B = [19 22; 43 50]
	res, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}

	expected := []float64{19, 22, 43, 50}
	for i, val := range res.Data {
		if val != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], val)
		}
	}
}

func TestAdd(t *testing.T) {
	a := NewTensor(2, 2)
	a.Data = []float64{1, 2, 3, 4}
	b := NewTensor(2, 2)
	b.Data = []float64{10, 20, 30, 40}

	res, err := Add(a, b)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	expected := []float64{11, 22, 33, 44}
	for i, val := range res.Data {
		if val != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], val)
		}
	}
}

func TestTranspose(t *testing.T) {
	// A = [1 2 3; 4 5 6] (2x3)
	a := NewTensor(2, 3)
	a.Data = []float64{1, 2, 3, 4, 5, 6}

	// Expected At = [1 4; 2 5; 3 6] (3x2)
	res := Transpose(a)

	if res.Rows != 3 || res.Cols != 2 {
		t.Fatalf("Expected 3x2, got %dx%d", res.Rows, res.Cols)
	}

	expected := []float64{1, 4, 2, 5, 3, 6}
	for i, val := range res.Data {
		if val != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], val)
		}
	}
}

func TestSoftmax(t *testing.T) {
	tnsr := NewTensor(1, 3)
	tnsr.Data = []float64{1.0, 2.0, 3.0}

	Softmax(tnsr)

	sum := 0.0
	for _, val := range tnsr.Data {
		sum += val
	}

	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("Expected sum 1.0, got %f", sum)
	}

	// Verify relative order is preserved
	if !(tnsr.Data[0] < tnsr.Data[1] && tnsr.Data[1] < tnsr.Data[2]) {
		t.Errorf("Softmax did not preserve relative order: %v", tnsr.Data)
	}
}

func TestLayerNorm(t *testing.T) {
	tnsr := NewTensor(1, 4)
	tnsr.Data = []float64{10, 20, 30, 40}

	LayerNorm(tnsr)

	mean, variance := tnsr.RowMeanVar(0)

	if math.Abs(mean) > 1e-9 {
		t.Errorf("Expected mean 0, got %f", mean)
	}
	if math.Abs(variance-1.0) > 1e-6 {
		t.Errorf("Expected variance 1, got %f", variance)
	}
}
