package tokenizer

import (
	"testing"
)

func TestTokenizer(t *testing.T) {
	text := "абв abc"
	tk := NewTokenizer(text)

	if tk.VocabSize() != 7 { // 'а', 'б', 'в', ' ', 'a', 'b', 'c'
		t.Errorf("Expected vocab size 7, got %d", tk.VocabSize())
	}

	input := "баб"
	encoded := tk.Encode(input)
	decoded := tk.Decode(encoded)

	if input != decoded {
		t.Errorf("Expected %s, got %s", input, decoded)
	}
}
