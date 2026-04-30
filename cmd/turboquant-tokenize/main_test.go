package main

import "testing"

func TestSplitTokensChunksAndLimits(t *testing.T) {
	got := splitTokens([]int{1, 2, 3, 4, 5, 6, 7}, 3, 2)
	if len(got) != 2 {
		t.Fatalf("len(splitTokens) = %d, want 2", len(got))
	}
	if got[0].ID != "seq-00000" || got[1].ID != "seq-00001" {
		t.Fatalf("ids = %q, %q", got[0].ID, got[1].ID)
	}
	if len(got[0].Tokens) != 3 || len(got[1].Tokens) != 3 {
		t.Fatalf("chunk lengths = %d, %d; want 3, 3", len(got[0].Tokens), len(got[1].Tokens))
	}
	if got[1].Tokens[0] != 4 {
		t.Fatalf("second chunk starts at %d, want 4", got[1].Tokens[0])
	}
}

func TestSplitTokensKeepsShortTailWhenScorable(t *testing.T) {
	got := splitTokens([]int{1, 2, 3, 4, 5}, 3, 4)
	if len(got) != 2 {
		t.Fatalf("len(splitTokens) = %d, want 2", len(got))
	}
	if len(got[1].Tokens) != 2 {
		t.Fatalf("tail length = %d, want 2", len(got[1].Tokens))
	}
}
