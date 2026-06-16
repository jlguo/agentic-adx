package ab

import (
	"testing"
)

// fakeExperimentManager creates an ExperimentManager with in-memory experiments,
// bypassing MySQL. Used by tests that don't need real DB connectivity.
func fakeExperimentManager(exps []*Experiment) *ExperimentManager {
	return &ExperimentManager{
		experiments: exps,
	}
}

func TestAssignExperiment_Deterministic(t *testing.T) {
	mgr := fakeExperimentManager([]*Experiment{
		{ID: 1, Name: "test-exp", TrafficRatio: 0.5, HashSalt: "salt-abc"},
	})

	// Same input must always produce the same variant.
	first := mgr.AssignExperiment("req-001")
	if first == nil {
		t.Fatal("expected assignment, got nil")
	}

	for i := 0; i < 100; i++ {
		again := mgr.AssignExperiment("req-001")
		if again.Variant != first.Variant {
			t.Errorf("hash not deterministic: first=%s, again=%s",
				first.Variant, again.Variant)
		}
		if again.ExperimentID != first.ExperimentID {
			t.Errorf("experiment ID changed: %d vs %d",
				first.ExperimentID, again.ExperimentID)
		}
	}
}

func TestAssignExperiment_TrafficRatio(t *testing.T) {
	mgr := fakeExperimentManager([]*Experiment{
		{ID: 1, Name: "test-exp", TrafficRatio: 0.5, HashSalt: "salt-xyz"},
	})

	treatment := 0
	control := 0
	const n = 200

	for i := 0; i < n; i++ {
		reqID := "req-" + string(rune('a'+i%26)) + string(rune('a'+(i/26)%26))
		assign := mgr.AssignExperiment(reqID)
		if assign == nil {
			t.Fatalf("expected assignment for %s", reqID)
		}
		switch assign.Variant {
		case "treatment":
			treatment++
		case "control":
			control++
		default:
			t.Fatalf("unexpected variant: %s", assign.Variant)
		}
	}

	if treatment+control != n {
		t.Errorf("expected %d assignments, got %d", n, treatment+control)
	}

	// With ratio=0.5 and 100 hash buckets, expect roughly 50/50 split.
	// Allow ±20% tolerance due to small sample.
	ratio := float64(treatment) / float64(n)
	if ratio < 0.35 || ratio > 0.65 {
		t.Errorf("traffic ratio out of range: treatment=%.2f (wanted ~0.5)", ratio)
	}

	t.Logf("treatment=%d control=%d (ratio=%.3f)", treatment, control, ratio)
}

func TestAssignExperiment_DifferentSalts(t *testing.T) {
	mgr := fakeExperimentManager([]*Experiment{
		{ID: 1, Name: "exp-a", TrafficRatio: 0.3, HashSalt: "salt-aaa"},
		{ID: 2, Name: "exp-b", TrafficRatio: 0.3, HashSalt: "salt-bbb"},
	})

	// With first-match-wins, all requests are claimed by exp-a.
	// We verify that only exp-a gets assignments.
	for i := 0; i < 50; i++ {
		reqID := "req-" + string(rune('a'+i%26)) + string(rune('0'+i%10))
		assign := mgr.AssignExperiment(reqID)
		if assign == nil {
			t.Fatalf("expected assignment for %s", reqID)
		}
		if assign.ExperimentID != 1 {
			t.Errorf("expected first experiment (id=1) to claim %s, got id=%d",
				reqID, assign.ExperimentID)
		}
	}
}

func TestAssignExperiment_NoActiveExperiments(t *testing.T) {
	mgr := fakeExperimentManager(nil)

	assign := mgr.AssignExperiment("req-001")
	if assign != nil {
		t.Errorf("expected nil with no experiments, got %+v", assign)
	}
}

func TestAssignExperiment_ZeroTrafficRatio(t *testing.T) {
	mgr := fakeExperimentManager([]*Experiment{
		{ID: 1, Name: "zero-exp", TrafficRatio: 0.0, HashSalt: "salt-zero"},
	})

	// All traffic should go to control when ratio=0.
	for i := 0; i < 10; i++ {
		assign := mgr.AssignExperiment("req-" + string(rune('a'+i)))
		if assign == nil {
			t.Fatal("expected assignment")
		}
		if assign.Variant != "control" {
			t.Errorf("expected control at ratio=0, got %s", assign.Variant)
		}
	}
}

func TestAssignExperiment_FullTrafficRatio(t *testing.T) {
	mgr := fakeExperimentManager([]*Experiment{
		{ID: 1, Name: "full-exp", TrafficRatio: 1.0, HashSalt: "salt-full"},
	})

	// All traffic should go to treatment when ratio=1.
	for i := 0; i < 10; i++ {
		assign := mgr.AssignExperiment("req-" + string(rune('a'+i)))
		if assign == nil {
			t.Fatal("expected assignment")
		}
		if assign.Variant != "treatment" {
			t.Errorf("expected treatment at ratio=1, got %s", assign.Variant)
		}
	}
}

func TestGetExperiment(t *testing.T) {
	mgr := fakeExperimentManager([]*Experiment{
		{ID: 1, Name: "exp-a", HashSalt: "salt-a"},
		{ID: 2, Name: "exp-b", HashSalt: "salt-b"},
	})

	if exp := mgr.GetExperiment(1); exp == nil || exp.Name != "exp-a" {
		t.Error("GetExperiment(1) failed")
	}
	if exp := mgr.GetExperiment(2); exp == nil || exp.Name != "exp-b" {
		t.Error("GetExperiment(2) failed")
	}
	if exp := mgr.GetExperiment(999); exp != nil {
		t.Error("GetExperiment(999) should return nil")
	}
}
