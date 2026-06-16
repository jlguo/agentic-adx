package pipeline

import (
	"context"
	"testing"

	"github.com/agentic-adx/adx/internal/model"
)

func TestExtractUserVector(t *testing.T) {
	req := &model.BidRequest{
		ID: "test-1",
		Device: &model.Device{
			UA: "Mozilla/5.0",
			IP: "1.2.3.4",
		},
	}
	vec := extractUserVector(req)
	if len(vec) != 384 {
		t.Errorf("expected 384-dim vector, got %d", len(vec))
	}
}

func TestBuildUserContext(t *testing.T) {
	req := &model.BidRequest{
		Device: &model.Device{
			UA: "Mozilla/5.0 Chrome",
			Geo: &model.Geo{Country: "US"},
		},
		Site: &model.Site{Domain: "example.com"},
	}
	ctx := buildUserContext(req)
	if ctx == "" {
		t.Error("expected non-empty user context")
	}
	if !contains(ctx, "example.com") {
		t.Error("user context should contain site domain")
	}
	if !contains(ctx, "US") {
		t.Error("user context should contain country")
	}
}

func TestPipeline_NoDependencies(t *testing.T) {
	p := NewPipeline(nil, nil)
	if p == nil {
		t.Fatal("pipeline should not be nil")
	}
	if p.lrScorer == nil {
		t.Fatal("LR scorer should be initialized")
	}
}

func TestPipelineProcess_EmptyContext(t *testing.T) {
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
