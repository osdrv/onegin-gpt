package data

import "math/rand"

type Dataset struct {
	Tokens []int
}

func NewDataset(tokens []int) *Dataset {
	return &Dataset{Tokens: tokens}
}

// Returns two 2D slices: X(inputs) and Y(targets)
// X shape: [batchSize, contextLen]
// Y shape: [batchSize, contextLen]
func (d *Dataset) GetBatch(batchSize, contextLen int) ([][]int, [][]int) {
	x := make([][]int, batchSize)
	y := make([][]int, batchSize)

	for i := range batchSize {
		// pick a random starting point
		start := rand.Intn(len(d.Tokens) - contextLen - 1)
		x[i] = d.Tokens[start : start+contextLen]
		y[i] = d.Tokens[start+1 : start+contextLen+1]
	}

	return x, y
}
