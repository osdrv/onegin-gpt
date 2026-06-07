package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"mini-gpt/internal/tensor"
)

type Reader struct {
	r          io.ReadSeeker
	tensors    map[string]TensorInfo
	metadata   map[string]any
	dataOffset int64
}

func NewReader(r io.ReadSeeker) *Reader {
	return &Reader{
		r:        r,
		tensors:  make(map[string]TensorInfo),
		metadata: make(map[string]any),
	}
}

// Helpers to read GGUF types
func (gr *Reader) readUint32() (uint32, error) {
	var v uint32
	err := binary.Read(gr.r, binary.LittleEndian, &v)
	return v, err
}

func (gr *Reader) readUint64() (uint64, error) {
	var v uint64
	err := binary.Read(gr.r, binary.LittleEndian, &v)
	return v, err
}

func (gr *Reader) readString() (string, error) {
	n, err := gr.readUint64()
	if err != nil {
		return "", err
	}
	buf := make([]byte, n)
	_, err = io.ReadFull(gr.r, buf)
	return string(buf), err
}

func (gr *Reader) ReadAll() error {
	// 1. Verify Magic & Version
	magic, _ := gr.readUint32()
	if magic != GGUFMagic {
		return fmt.Errorf("invalid GGUF magic")
	}
	version, _ := gr.readUint32()
	if version != GGUFVersion {
		return fmt.Errorf("unsupported GGUF version")
	}

	// 2. Read Counts
	numTensors, _ := gr.readUint64()
	numMetadata, _ := gr.readUint64()

	// 3. Load Metadata (Strings only for now)
	for i := uint64(0); i < numMetadata; i++ {
		key, _ := gr.readString()
		vType, _ := gr.readUint32()
		if GGUFValueType(vType) == TypeString {
			val, _ := gr.readString()
			gr.metadata[key] = val
		}
	}

	// 4. Load Tensor Info Index
	for i := uint64(0); i < numTensors; i++ {
		name, _ := gr.readString()
		nDims, _ := gr.readUint32()
		dims := make([]uint64, nDims)
		for d := uint32(0); d < nDims; d++ {
			dims[d], _ = gr.readUint64()
		}
		tType, _ := gr.readUint32()
		offset, _ := gr.readUint64()

		gr.tensors[name] = TensorInfo{
			Name: name, Dimensions: dims, Type: tType, Offset: offset,
		}
	}

	// 5. Mark the start of the data section (aligned to 32 bytes)
	pos, _ := gr.r.Seek(0, io.SeekCurrent)
	gr.dataOffset = (pos + int64(Alignment-1)) & ^int64(Alignment-1)

	return nil
}

func (gr *Reader) LoadTensor(name string) (*tensor.Tensor, error) {
	info, ok := gr.tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %s not found", name)
	}

	// Jump to tensor data
	_, err := gr.r.Seek(gr.dataOffset+int64(info.Offset), io.SeekStart)
	if err != nil {
		return nil, err
	}

	// GGUF dims are [Cols, Rows, Batch]
	cols := int(info.Dimensions[0])
	rows := 1
	if len(info.Dimensions) > 1 {
		rows = int(info.Dimensions[1])
	}
	batch := 1
	if len(info.Dimensions) > 2 {
		batch = int(info.Dimensions[2])
	}

	t := tensor.NewTensor(batch, rows, cols)
	for i := range t.Data {
		var f32 float32
		binary.Read(gr.r, binary.LittleEndian, &f32)
		t.Data[i] = float64(f32)
	}
	return t, nil
}
