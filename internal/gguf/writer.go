package gguf

import (
	"encoding/binary"
	"io"
)

const (
	GGUFMagic   = 0x46554747
	GGUFVersion = 3
	Alignment   = 32
)

type GGUFValueType uint32

const (
	TypeUint8   GGUFValueType = 0
	TypeInt8    GGUFValueType = 1
	TypeUint16  GGUFValueType = 2
	TypeInt16   GGUFValueType = 3
	TypeUint32  GGUFValueType = 4
	TypeInt32   GGUFValueType = 5
	TypeFloat32 GGUFValueType = 6
	TypeBool    GGUFValueType = 7
	TypeString  GGUFValueType = 8
	TypeArray   GGUFValueType = 9
)

type TensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       uint32
	Offset     uint64
	Size       uint64
}

type Writer struct {
	w          io.Writer
	tensors    []TensorInfo
	metadata   map[string]any
	currentPos int64
}

func NewWriter(w io.Writer) *Writer {
	return &Writer{
		w:        w,
		metadata: make(map[string]any),
	}
}

func (gw *Writer) write(data any) error {
	if s, ok := data.(string); ok {
		if err := gw.write(uint64(len(s))); err != nil {
			return err
		}
		n, err := gw.w.Write([]byte(s))
		gw.currentPos += int64(n)
		return err
	}
	err := binary.Write(gw.w, binary.LittleEndian, data)
	if err == nil {
		gw.currentPos += int64(binary.Size(data))
	}
	return err
}

func (gw *Writer) WriteHeaders() error {
	if err := gw.write(uint32(GGUFMagic)); err != nil {
		return err
	}
	if err := gw.write(uint32(GGUFVersion)); err != nil {
		return err
	}

	if err := gw.write(uint64(len(gw.tensors))); err != nil {
		return err
	}

	if err := gw.write(uint64(len(gw.metadata))); err != nil {
		return err
	}

	for k, v := range gw.metadata {
		if err := gw.write(k); err != nil {
			return err
		}
		if err := gw.write(uint32(TypeString)); err != nil {
			return err
		}
		if err := gw.write(v.(string)); err != nil {
			return err
		}
	}

	for _, t := range gw.tensors {
		if err := gw.write(t.Name); err != nil {
			return err
		}
		if err := gw.write(uint32(len(t.Dimensions))); err != nil {
			return err
		}
		for _, dim := range t.Dimensions {
			if err := gw.write(uint64(dim)); err != nil {
				return err
			}
		}
		// 0 == FP32
		if err := gw.write(uint32(0)); err != nil {
			return err
		}
		if err := gw.write(uint64(t.Offset)); err != nil {
			return err
		}
	}
	return nil
}

func (gw *Writer) WritePadding() error {
	pad := (Alignment - (gw.currentPos % Alignment)) % Alignment
	if pad > 0 {
		_, err := gw.w.Write(make([]byte, pad))
		gw.currentPos += int64(pad)
		return err
	}
	return nil
}

func (gw *Writer) AddTensor(name string, dims []uint64, data []float64) {
	// GGUF uses Float32 (4 bytes)
	size := uint64(len(data) * 4)

	offset := uint64(0)
	if len(gw.tensors) > 0 {
		last := gw.tensors[len(gw.tensors)-1]
		offset = last.Offset + last.Size
		// Align to 32 bytes
		offset = (offset + uint64(Alignment-1)) & ^uint64(Alignment-1)
	}

	gw.tensors = append(gw.tensors, TensorInfo{
		Name:       name,
		Dimensions: dims,
		Size:       size,
		Offset:     offset,
	})
}

func (gw *Writer) WriteData(data []float64) error {
	for _, val := range data {
		if err := gw.write(float32(val)); err != nil {
			return err
		}
	}
	return gw.WritePadding()
}

func (gw *Writer) SetMetadta(key, value string) {
	gw.metadata[key] = value
}
