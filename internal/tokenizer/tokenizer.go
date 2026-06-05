package tokenizer

import "sort"

type Tokenizer struct {
	encoder   map[rune]int
	decoder   map[int]rune
	vocabSize int
}

func NewTokenizer(text string) *Tokenizer {
	runesMap := make(map[rune]bool)
	for _, r := range text {
		runesMap[r] = true
	}

	var uniqueRunes []rune
	for r := range runesMap {
		uniqueRunes = append(uniqueRunes, r)
	}

	sort.Slice(uniqueRunes, func(i, j int) bool {
		return uniqueRunes[i] < uniqueRunes[j]
	})

	vocabSize := len(uniqueRunes)
	encoder := make(map[rune]int, vocabSize)
	decoder := make(map[int]rune, vocabSize)

	for i, r := range uniqueRunes {
		encoder[r] = i
		decoder[i] = r
	}

	return &Tokenizer{
		encoder:   encoder,
		decoder:   decoder,
		vocabSize: vocabSize,
	}
}

func (t *Tokenizer) Encode(s string) []int {
	tokens := make([]int, 0, len(s))
	for _, r := range s {
		tokens = append(tokens, t.encoder[r])
	}
	return tokens
}

func (t *Tokenizer) Decode(tokens []int) string {
	runes := make([]rune, 0, len(tokens))
	for _, id := range tokens {
		if r, ok := t.decoder[id]; ok {
			runes = append(runes, r)
		}
	}
	return string(runes)
}

func (t *Tokenizer) VocabSize() int {
	return t.vocabSize
}
