package tokenizer

type BPE struct {
	vocab     map[int][]byte
	merges    map[[2]int]int
	vocabSize int
}

func (b *BPE) Train(text string, numMerges int) {
	b.vocab = make(map[int][]byte)
	b.merges = make(map[[2]int]int)

	// initialize vocab with bytes 0-255
	for i := range 256 {
		b.vocab[i] = []byte{byte(i)}
	}
	b.vocabSize = 256

	// convert the text to initial byte IDs
	tokens := make([]int, len(text))
	for i := range len(text) {
		tokens[i] = int(text[i])
	}

	// iteratively merge the most frequent pairs
	for range numMerges {
		stats := getStats(tokens)
		if len(stats) == 0 {
			break
		}

		var bestPair [2]int
		maxFreq := -1
		for pair, freq := range stats {
			if freq > maxFreq {
				maxFreq = freq
				bestPair = pair
			}
		}

		newId := b.vocabSize
		b.merges[bestPair] = newId

		combined := append([]byte{}, b.vocab[bestPair[0]]...)
		combined = append(combined, b.vocab[bestPair[1]]...)
		b.vocab[newId] = combined

		tokens = merge(tokens, bestPair, newId)
		b.vocabSize++
	}
}

func (b *BPE) Encode(text string) []int {
	tokens := make([]int, len(text))
	for i := range len(text) {
		tokens[i] = int(text[i])
	}

	for {
		stats := getStats(tokens)
		var pairToMerge *[2]int
		minMergeRank := b.vocabSize + 1

		for pair := range stats {
			if rank, ok := b.merges[pair]; ok {
				if rank < minMergeRank {
					minMergeRank = rank
					p := pair
					pairToMerge = &p
				}
			}
		}

		if pairToMerge == nil {
			break
		}

		tokens = merge(tokens, *pairToMerge, minMergeRank)
	}

	return tokens
}

func (b *BPE) Decode(ids []int) string {
	var res []byte
	for _, id := range ids {
		res = append(res, b.vocab[id]...)
	}
	return string(res)
}

func getStats(ids []int) map[[2]int]int {
	counts := make(map[[2]int]int)
	for i := range len(ids) - 1 {
		pair := [2]int{ids[i], ids[i+1]}
		counts[pair]++
	}
	return counts
}

func merge(ids []int, pair [2]int, newId int) []int {
	newIds := make([]int, 0, len(ids))
	for i := 0; i < len(ids); i++ {
		if i < len(ids)-1 && ids[i] == pair[0] && ids[i+1] == pair[1] {
			newIds = append(newIds, newId)
			i++
		} else {
			newIds = append(newIds, ids[i])
		}
	}
	return newIds
}
