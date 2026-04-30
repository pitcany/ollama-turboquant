package dumpindex

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadDirAndInferLayers(t *testing.T) {
	dir := t.TempDir()
	writeDump(t, dir, "000000_node_unknown_src0_RESHAPE_f32", map[string]string{
		"seq": "0", "node": "-1", "tag": "src0", "op": "RESHAPE", "type": "f32", "name": "k",
		"nbytes": "1048576", "ne": "512,512,1,1", "nb": "4,2048,1048576,1048576", "op_params": "0,0,0,0,0,0,0,0",
	})
	writeDump(t, dir, "000001_node_unknown_out_SET_ROWS_turbo4", map[string]string{
		"seq": "1", "node": "-1", "tag": "out", "op": "SET_ROWS", "type": "turbo4", "name": "kpacked",
		"nbytes": "278528", "ne": "512,1024,1,1", "nb": "68,272,278528,278528", "op_params": "0,0,0,0,0,0,0,0",
	})
	writeDump(t, dir, "000002_node_unknown_src0_RESHAPE_f32", map[string]string{
		"seq": "2", "node": "-1", "tag": "src0", "op": "RESHAPE", "type": "f32", "name": "v",
		"nbytes": "1048576", "ne": "512,512,1,1", "nb": "4,2048,1048576,1048576", "op_params": "0,0,0,0,0,0,0,0",
	})
	writeDump(t, dir, "000003_node_unknown_out_SET_ROWS_turbo4", map[string]string{
		"seq": "3", "node": "-1", "tag": "out", "op": "SET_ROWS", "type": "turbo4", "name": "vpacked",
		"nbytes": "278528", "ne": "512,1024,1,1", "nb": "68,272,278528,278528", "op_params": "0,0,0,0,0,0,0,0",
	})
	writeDump(t, dir, "000004_node_unknown_out_TURBO_WHT_f32", map[string]string{
		"seq": "4", "node": "-1", "tag": "out", "op": "TURBO_WHT", "type": "f32", "name": "q",
		"nbytes": "7340032", "ne": "128,28,512,1", "nb": "4,512,14336,7340032", "op_params": "0,0,0,0,128,0,0,0",
	})
	writeDump(t, dir, "000005_node_unknown_out_FLASH_ATTN_EXT_f32", map[string]string{
		"seq": "5", "node": "-1", "tag": "out", "op": "FLASH_ATTN_EXT", "type": "f32", "name": "fa",
		"nbytes": "7340032", "ne": "128,28,512,1", "nb": "4,512,14336,7340032", "op_params": "1035273459,0,0,10,0,0,0,0",
	})
	writeDump(t, dir, "000006_node_unknown_out_TURBO_WHT_f32", map[string]string{
		"seq": "6", "node": "-1", "tag": "out", "op": "TURBO_WHT", "type": "f32", "name": "inv",
		"nbytes": "7340032", "ne": "128,28,512,1", "nb": "4,512,14336,7340032", "op_params": "1,0,0,0,128,0,0,0",
	})

	records, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	layers := InferLayers(records)
	if len(layers) != 1 {
		t.Fatalf("layers=%d, want 1", len(layers))
	}
	summaries := Summaries(layers)
	got := summaries[0]
	if !got.Complete {
		t.Fatalf("layer incomplete: %s", got.Incomplete)
	}
	if got.KRows != 2048 || got.KVHeads != 4 || got.QHeads != 28 || got.QTokens != 512 {
		t.Fatalf("bad dims: %+v", got)
	}
	if got.KSeq != 0 || got.VSeq != 2 || got.QSeq != 4 {
		t.Fatalf("bad seqs: %+v", got)
	}
	if got.KPacked == "" || got.VPacked == "" || got.Flash == "" || got.InvWHT == "" {
		t.Fatalf("missing paired records: %+v", got)
	}
}

func TestWriteCSV(t *testing.T) {
	layers := []LayerSummary{{
		Layer:    0,
		Complete: true,
		KRows:    2048,
		KVHeads:  4,
		QHeads:   28,
		QTokens:  512,
		KSrc:     "/tmp/k.bin",
		VSrc:     "/tmp/v.bin",
		QRot:     "/tmp/q.bin",
	}}
	var out bytes.Buffer
	if err := WriteCSV(&out, layers); err != nil {
		t.Fatalf("WriteCSV: %v", err)
	}
	text := out.String()
	for _, want := range []string{"layer,complete", "2048,4,28,512", "/tmp/k.bin,/tmp/v.bin,/tmp/q.bin"} {
		if !strings.Contains(text, want) {
			t.Fatalf("CSV missing %q in:\n%s", want, text)
		}
	}
}

func writeDump(t *testing.T, dir, base string, values map[string]string) {
	t.Helper()
	txt := filepath.Join(dir, base+".txt")
	bin := filepath.Join(dir, base+".bin")
	var b strings.Builder
	keys := []string{"seq", "node", "tag", "op", "type", "name", "nbytes", "ne", "nb", "op_params"}
	for _, key := range keys {
		b.WriteString(key)
		b.WriteByte('=')
		b.WriteString(values[key])
		b.WriteByte('\n')
	}
	if err := os.WriteFile(txt, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write txt: %v", err)
	}
	nbytes := 0
	if values["nbytes"] != "" {
		var err error
		nbytes64, err := parseInt64(values["nbytes"])
		if err != nil {
			t.Fatalf("parse nbytes: %v", err)
		}
		nbytes = int(nbytes64)
	}
	if err := os.WriteFile(bin, make([]byte, nbytes), 0o644); err != nil {
		t.Fatalf("write bin: %v", err)
	}
}
