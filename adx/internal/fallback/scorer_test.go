package fallback

import (
	"testing"

	"github.com/agentic-adx/adx/internal/model"
)

func TestLRScorer_Score(t *testing.T) {
	scorer := NewLRScorer()
	bid := &model.Bid{
		ID:    "bid-1",
		Price: 3.00,
		AdID:  "ad-1",
	}
	score := scorer.Score(bid)
	if score <= 0 {
		t.Errorf("expected positive score, got %.4f", score)
	}
}

func TestLRScorer_ScoreWithECPM(t *testing.T) {
	scorer := NewLRScorer()
	bid := &model.Bid{
		ID:    "bid-1",
		Price: 2.00,
	}
	ecpm := scorer.ScoreWithECPM(bid)
	if ecpm <= 0 {
		t.Errorf("expected positive eCPM, got %.4f", ecpm)
	}
}

func TestLRScorer_SameSeedSameFirstScore(t *testing.T) {
	bid := &model.Bid{ID: "bid-1", Price: 5.00}
	s1 := NewLRScorer().Score(bid)
	s2 := NewLRScorer().Score(bid)
	if s1 != s2 {
		t.Errorf("same seed should produce same first score: %.4f vs %.4f", s1, s2)
	}
}

func TestLRScorer_HigherBidHigherScore(t *testing.T) {
	scorer := NewLRScorer()
	low := &model.Bid{ID: "low", Price: 1.00}
	high := &model.Bid{ID: "high", Price: 10.00}
	lowScore := scorer.ScoreWithECPM(low)
	highScore := scorer.ScoreWithECPM(high)
	if lowScore >= highScore {
		t.Errorf("higher bid should produce higher eCPM: low=%.4f high=%.4f", lowScore, highScore)
	}
}

func TestScoreMultiple(t *testing.T) {
	scorer := NewLRScorer()
	bids := []model.Bid{
		{ID: "a", Price: 1.0},
		{ID: "b", Price: 2.0},
		{ID: "c", Price: 3.0},
	}
	scores := ScoreMultiple(scorer, bids)
	if len(scores) != 3 {
		t.Errorf("expected 3 scores, got %d", len(scores))
	}
	for _, bid := range bids {
		if _, ok := scores[bid.ID]; !ok {
			t.Errorf("missing score for bid %s", bid.ID)
		}
	}
}
