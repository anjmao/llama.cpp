package rebalancer

import (
	"testing"
	"time"
)

// ---------- Counter tests ----------

func TestCounterUpdateAndRank(t *testing.T) {
	c := NewCounter(2, 4, 30*time.Second)

	// Layer 0: expert 1 selected 3 times, expert 0 once
	c.Update(0, []int{1})
	c.Update(0, []int{1})
	c.Update(0, []int{1})
	c.Update(0, []int{0})

	ranked := c.Rank(0)
	if ranked[0].ID != 1 {
		t.Errorf("expected expert 1 first, got %d (score %v)", ranked[0].ID, ranked[0].Score)
	}
	if ranked[1].ID != 0 {
		t.Errorf("expected expert 0 second, got %d", ranked[1].ID)
	}
	// Experts 2 and 3 should be last with score 0
	if ranked[2].Score != 0 || ranked[3].Score != 0 {
		t.Errorf("expected zero scores for unused experts")
	}
}

func TestCounterDecay(t *testing.T) {
	c := NewCounter(1, 4, 100*time.Millisecond)

	c.Update(0, []int{0})
	score1 := c.Rank(0)[0].Score
	if score1 == 0 {
		t.Fatal("expected non-zero score after update")
	}

	time.Sleep(350 * time.Millisecond)
	ranked := c.Rank(0)
	// After ~3.5 tau, score should be heavily decayed (factor ~0.15 or less)
	if ranked[0].Score > score1*0.2 {
		t.Errorf("expected heavy decay, got score %v vs initial %v", ranked[0].Score, score1)
	}
}

func TestCounterRankEmpty(t *testing.T) {
	c := NewCounter(2, 4, 30*time.Second)
	ranked := c.Rank(0)
	if len(ranked) != 4 {
		t.Fatalf("expected 4 entries, got %d", len(ranked))
	}
	for _, s := range ranked {
		if s.Score != 0 {
			t.Errorf("expected all zero scores, got %v", s.Score)
		}
	}
}

func TestCounterInvalidLayer(t *testing.T) {
	c := NewCounter(2, 4, 30*time.Second)
	c.Update(5, []int{0}) // out of range, should be ignored
	if ranked := c.Rank(5); ranked != nil {
		t.Errorf("expected nil for out-of-range layer, got %v", ranked)
	}
}

func TestCounterSnapshot(t *testing.T) {
	c := NewCounter(1, 3, 30*time.Second)
	c.Update(0, []int{0, 1})
	snap := c.Snapshot()
	if len(snap) != 1 || len(snap[0]) != 3 {
		t.Fatalf("unexpected snapshot shape: %v", snap)
	}
	if snap[0][0] == 0 || snap[0][1] == 0 {
		t.Errorf("expected non-zero scores for experts 0 and 1")
	}
	if snap[0][2] != 0 {
		t.Errorf("expected zero score for expert 2")
	}
}

// ---------- Rebalancer tests ----------

func TestComputeTargetBasicRanking(t *testing.T) {
	r := New(2, 3, 1)

	// Experts ranked: 2 > 0 > 3 > 1 (by score)
	ranked := []ExpertScore{
		{ID: 2, Score: 10},
		{ID: 0, Score: 8},
		{ID: 3, Score: 5},
		{ID: 1, Score: 1},
	}

	// No current placement -> should pick top-3: {0, 2, 3}
	target := r.ComputeTarget(0, ranked, nil)
	if len(target) != 3 {
		t.Fatalf("expected 3 experts, got %d", len(target))
	}
	expected := []int{0, 2, 3}
	for i, e := range target {
		if e != expected[i] {
			t.Errorf("position %d: expected %d, got %d", i, expected[i], e)
		}
	}
}

func TestHysteresisKeepsBoundaryExpert(t *testing.T) {
	r := New(1, 2, 1)

	// Expert 1 is currently placed (GPU), but has dropped to rank 3 (below K=2)
	// but within K+hysteresis=3, so it should be kept.
	ranked := []ExpertScore{
		{ID: 2, Score: 10},
		{ID: 0, Score: 8},
		{ID: 1, Score: 5}, // rank 3, within K+hysteresis=3
		{ID: 3, Score: 1},
	}

	currentGPU := []int{1}
	target := r.ComputeTarget(0, ranked, currentGPU)

	// Expert 1 should still be in target (hysteresis keeps it)
	found := false
	for _, e := range target {
		if e == 1 {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected expert 1 to be retained by hysteresis, target: %v", target)
	}
}

func TestHysteresisDropsExpertBeyondThreshold(t *testing.T) {
	r := New(1, 2, 1)

	// Expert 1 is currently placed but has dropped to rank 4 (beyond K+hysteresis=3)
	ranked := []ExpertScore{
		{ID: 2, Score: 10},
		{ID: 0, Score: 8},
		{ID: 3, Score: 5},
		{ID: 1, Score: 1}, // rank 4, beyond K+hysteresis=3
	}

	currentGPU := []int{1}
	target := r.ComputeTarget(0, ranked, currentGPU)

	// Expert 1 should be dropped (rank 4 > K+hysteresis=3)
	for _, e := range target {
		if e == 1 {
			t.Errorf("expected expert 1 to be dropped, but it's in target: %v", target)
		}
	}
}

func TestDiffSkipsUnchangedLayers(t *testing.T) {
	nLayers := 4
	current := []LayerInfo{
		{Layer: 0, Dynamic: true, ExpertsGPU: []int{0, 1, 2}},
		{Layer: 1, Dynamic: true, ExpertsGPU: []int{10, 11, 12}},
		{Layer: 2, OnGPU: false}, // dense layer, no experts
	}

	// Layer 0 unchanged, layer 1 changed, layer 3 new
	target := map[int][]int{
		0: {0, 1, 2},   // same as current
		1: {10, 11, 13}, // different (12 -> 13)
		3: {30, 31, 32}, // new placement
	}

	changes := Diff(nLayers, current, target)
	if changes == nil {
		t.Fatal("expected non-nil changes")
	}
	if len(changes) != 2 {
		t.Fatalf("expected 2 changes (layers 1 and 3), got %d: %v", len(changes), changes)
	}

	// Layer 0 should be skipped
	for _, c := range changes {
		if c.Layer == 0 {
			t.Errorf("layer 0 should be skipped (unchanged)")
		}
	}
}

func TestDiffReturnsNilWhenNoChanges(t *testing.T) {
	nLayers := 2
	current := []LayerInfo{
		{Layer: 0, Dynamic: true, ExpertsGPU: []int{0, 1}},
		{Layer: 1, Dynamic: true, ExpertsGPU: []int{10, 11}},
	}
	target := map[int][]int{
		0: {0, 1},
		1: {10, 11},
	}

	changes := Diff(nLayers, current, target)
	if changes != nil {
		t.Errorf("expected nil when nothing changed, got %v", changes)
	}
}

func TestComputeTargetInvalidLayer(t *testing.T) {
	r := New(2, 3, 1)
	ranked := []ExpertScore{{ID: 0, Score: 1}}
	if target := r.ComputeTarget(5, ranked, nil); target != nil {
		t.Errorf("expected nil for invalid layer, got %v", target)
	}
}

func TestComputeTargetFillsToK(t *testing.T) {
	r := New(1, 4, 1)
	ranked := []ExpertScore{
		{ID: 0, Score: 10},
		{ID: 1, Score: 8},
		{ID: 2, Score: 5},
		{ID: 3, Score: 1},
		{ID: 4, Score: 0},
	}
	// Start with one expert placed, should fill to 4
	target := r.ComputeTarget(0, ranked, []int{4})
	if len(target) != 4 {
		t.Fatalf("expected 4 experts, got %d: %v", len(target), target)
	}
}
