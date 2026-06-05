package main

import (
	"fmt"
	"mini-gpt/internal/tokenizer"
	"os"
)

func main() {
	data, _ := os.ReadFile("data/onegin.txt")
	text := string(data)

	tok := tokenizer.NewTokenizer(text)
	sample := "привет, мир"
	encoded := tok.Encode(sample)
	decoded := tok.Decode(encoded)

	fmt.Printf("Original: %s\n", sample)
	fmt.Printf("Encoded: %+v\n", encoded)
	fmt.Printf("Decoded: %+v\n", decoded)
}
