package dumpindex

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type Record struct {
	Seq      int      `json:"seq"`
	Node     int      `json:"node"`
	Tag      string   `json:"tag"`
	Op       string   `json:"op"`
	Type     string   `json:"type"`
	Name     string   `json:"name"`
	NBytes   int64    `json:"nbytes"`
	NE       [4]int64 `json:"ne"`
	NB       [4]int64 `json:"nb"`
	OpParams [8]int32 `json:"op_params"`
	MetaPath string   `json:"meta_path"`
	BinPath  string   `json:"bin_path"`
}

type Layer struct {
	Layer   int     `json:"layer"`
	KSrc    *Record `json:"k_src,omitempty"`
	VSrc    *Record `json:"v_src,omitempty"`
	QRot    *Record `json:"q_rot,omitempty"`
	Flash   *Record `json:"flash,omitempty"`
	InvWHT  *Record `json:"inv_wht,omitempty"`
	KPacked *Record `json:"k_packed,omitempty"`
	VPacked *Record `json:"v_packed,omitempty"`
}

type LayerSummary struct {
	Layer      int    `json:"layer"`
	KSrc       string `json:"k_src,omitempty"`
	VSrc       string `json:"v_src,omitempty"`
	QRot       string `json:"q_rot,omitempty"`
	Flash      string `json:"flash,omitempty"`
	InvWHT     string `json:"inv_wht,omitempty"`
	KPacked    string `json:"k_packed,omitempty"`
	VPacked    string `json:"v_packed,omitempty"`
	KRows      int64  `json:"k_rows,omitempty"`
	QHeads     int64  `json:"q_heads,omitempty"`
	QTokens    int64  `json:"q_tokens,omitempty"`
	KVHeads    int64  `json:"kv_heads,omitempty"`
	KSeq       int    `json:"k_seq,omitempty"`
	VSeq       int    `json:"v_seq,omitempty"`
	QSeq       int    `json:"q_seq,omitempty"`
	Complete   bool   `json:"complete"`
	Incomplete string `json:"incomplete,omitempty"`
}

func LoadDir(dir string) ([]Record, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var records []Record
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".txt" {
			continue
		}
		record, err := ParseMetaFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			return nil, err
		}
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool {
		return records[i].Seq < records[j].Seq
	})
	return records, nil
}

func ParseMetaFile(path string) (Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return Record{}, err
	}
	defer f.Close()

	record := Record{MetaPath: path}
	values := map[string]string{}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		values[key] = value
	}
	if err := scanner.Err(); err != nil {
		return Record{}, err
	}

	var parseErr error
	record.Seq, parseErr = parseInt(values["seq"])
	if parseErr != nil {
		return Record{}, fmt.Errorf("%s: parse seq: %w", path, parseErr)
	}
	record.Node, _ = parseInt(values["node"])
	record.Tag = values["tag"]
	record.Op = values["op"]
	record.Type = values["type"]
	record.Name = values["name"]
	record.NBytes, _ = parseInt64(values["nbytes"])
	record.NE, _ = parseInt64List4(values["ne"])
	record.NB, _ = parseInt64List4(values["nb"])
	record.OpParams, _ = parseInt32List8(values["op_params"])
	record.BinPath = strings.TrimSuffix(path, filepath.Ext(path)) + ".bin"

	if record.NBytes > 0 {
		info, err := os.Stat(record.BinPath)
		if err != nil {
			return Record{}, fmt.Errorf("%s: missing bin: %w", path, err)
		}
		if info.Size() != record.NBytes {
			return Record{}, fmt.Errorf("%s: bin size %d != metadata nbytes %d", path, info.Size(), record.NBytes)
		}
	}
	return record, nil
}

func InferLayers(records []Record) []Layer {
	sorted := append([]Record(nil), records...)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Seq < sorted[j].Seq
	})

	var layers []Layer
	var pendingSrc []*Record
	var pendingPacked []*Record
	var current *Layer
	for i := range sorted {
		record := &sorted[i]
		switch {
		case record.IsSetRowsSrcF32():
			pendingSrc = append(pendingSrc, record)
		case record.IsPackedSetRows():
			pendingPacked = append(pendingPacked, record)
		case record.IsForwardQWHT():
			layer := Layer{Layer: len(layers), QRot: record}
			if len(pendingSrc) >= 2 {
				layer.KSrc = pendingSrc[len(pendingSrc)-2]
				layer.VSrc = pendingSrc[len(pendingSrc)-1]
			} else if len(pendingSrc) == 1 {
				layer.KSrc = pendingSrc[0]
			}
			if len(pendingPacked) >= 2 {
				layer.KPacked = pendingPacked[len(pendingPacked)-2]
				layer.VPacked = pendingPacked[len(pendingPacked)-1]
			} else if len(pendingPacked) == 1 {
				layer.KPacked = pendingPacked[0]
			}
			pendingSrc = nil
			pendingPacked = nil
			layers = append(layers, layer)
			current = &layers[len(layers)-1]
		case record.IsFlashAttention():
			if current != nil && current.Flash == nil {
				current.Flash = record
			}
		case record.IsInverseWHT():
			if current != nil && current.InvWHT == nil {
				current.InvWHT = record
			}
		}
	}
	return layers
}

func (r Record) IsSetRowsSrcF32() bool {
	return r.Tag == "src0" && r.Op == "RESHAPE" && r.Type == "f32" && r.NE[0]%128 == 0
}

func (r Record) IsPackedSetRows() bool {
	return r.Tag == "out" && r.Op == "SET_ROWS" && strings.HasPrefix(r.Type, "turbo")
}

func (r Record) IsForwardQWHT() bool {
	return r.Tag == "out" && r.Op == "TURBO_WHT" && r.Type == "f32" && r.OpParams[0] == 0 && r.NE[0] == 128
}

func (r Record) IsInverseWHT() bool {
	return r.Tag == "out" && r.Op == "TURBO_WHT" && r.Type == "f32" && r.OpParams[0] == 1 && r.NE[0] == 128
}

func (r Record) IsFlashAttention() bool {
	return r.Tag == "out" && r.Op == "FLASH_ATTN_EXT" && r.Type == "f32"
}

func Summaries(layers []Layer) []LayerSummary {
	out := make([]LayerSummary, 0, len(layers))
	for _, layer := range layers {
		s := LayerSummary{Layer: layer.Layer, Complete: true, KSeq: -1, VSeq: -1, QSeq: -1}
		if layer.KSrc != nil {
			s.KSrc = layer.KSrc.BinPath
			s.KSeq = layer.KSrc.Seq
			if layer.KSrc.NE[0] > 0 {
				s.KVHeads = layer.KSrc.NE[0] / 128
				s.KRows = layer.KSrc.NE[0] * layer.KSrc.NE[1] / 128
			}
		}
		if layer.VSrc != nil {
			s.VSrc = layer.VSrc.BinPath
			s.VSeq = layer.VSrc.Seq
		}
		if layer.QRot != nil {
			s.QRot = layer.QRot.BinPath
			s.QSeq = layer.QRot.Seq
			s.QHeads = layer.QRot.NE[1]
			s.QTokens = layer.QRot.NE[2]
		}
		if layer.Flash != nil {
			s.Flash = layer.Flash.BinPath
		}
		if layer.InvWHT != nil {
			s.InvWHT = layer.InvWHT.BinPath
		}
		if layer.KPacked != nil {
			s.KPacked = layer.KPacked.BinPath
		}
		if layer.VPacked != nil {
			s.VPacked = layer.VPacked.BinPath
		}
		var missing []string
		if s.KSrc == "" {
			missing = append(missing, "k_src")
		}
		if s.VSrc == "" {
			missing = append(missing, "v_src")
		}
		if s.QRot == "" {
			missing = append(missing, "q_rot")
		}
		if len(missing) > 0 {
			s.Complete = false
			s.Incomplete = strings.Join(missing, "|")
		}
		out = append(out, s)
	}
	return out
}

func WriteJSON(w io.Writer, layers []LayerSummary) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(layers)
}

func WriteCSV(w io.Writer, layers []LayerSummary) error {
	cw := csv.NewWriter(w)
	header := []string{
		"layer", "complete", "incomplete", "k_seq", "v_seq", "q_seq",
		"k_rows", "kv_heads", "q_heads", "q_tokens",
		"k_src", "v_src", "q_rot", "flash", "inv_wht", "k_packed", "v_packed",
	}
	if err := cw.Write(header); err != nil {
		return err
	}
	for _, layer := range layers {
		row := []string{
			strconv.Itoa(layer.Layer),
			strconv.FormatBool(layer.Complete),
			layer.Incomplete,
			intString(layer.KSeq),
			intString(layer.VSeq),
			intString(layer.QSeq),
			int64String(layer.KRows),
			int64String(layer.KVHeads),
			int64String(layer.QHeads),
			int64String(layer.QTokens),
			layer.KSrc,
			layer.VSrc,
			layer.QRot,
			layer.Flash,
			layer.InvWHT,
			layer.KPacked,
			layer.VPacked,
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}

func parseInt(s string) (int, error) {
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0, err
	}
	return v, nil
}

func parseInt64(s string) (int64, error) {
	v, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return 0, err
	}
	return v, nil
}

func parseInt64List4(s string) ([4]int64, error) {
	var out [4]int64
	parts := strings.Split(s, ",")
	if len(parts) != 4 {
		return out, fmt.Errorf("expected 4 values")
	}
	for i, part := range parts {
		v, err := parseInt64(part)
		if err != nil {
			return out, err
		}
		out[i] = v
	}
	return out, nil
}

func parseInt32List8(s string) ([8]int32, error) {
	var out [8]int32
	parts := strings.Split(s, ",")
	if len(parts) != 8 {
		return out, fmt.Errorf("expected 8 values")
	}
	for i, part := range parts {
		v, err := parseInt64(part)
		if err != nil {
			return out, err
		}
		out[i] = int32(v)
	}
	return out, nil
}

func intString(v int) string {
	if v < 0 {
		return ""
	}
	return strconv.Itoa(v)
}

func int64String(v int64) string {
	if v == 0 {
		return ""
	}
	return strconv.FormatInt(v, 10)
}
