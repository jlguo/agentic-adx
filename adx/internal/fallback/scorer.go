package fallback

import (
	"math"
	"math/rand"

	"github.com/agentic-adx/adx/internal/model"
)

type LRScorer struct {
	weights map[string]float64
	bias    float64
	rng     *rand.Rand
}

func NewLRScorer() *LRScorer {
	return &LRScorer{
		weights: map[string]float64{
			"bid_price":  0.40,
			"relevance":  0.30,
			"ctr_base":   0.15,
			"cvr_base":   0.10,
			"freshness":  0.05,
		},
		bias: 0.0,
		rng:  rand.New(rand.NewSource(42)),
	}
}

func (s *LRScorer) Score(bid *model.Bid) float64 {
	if bid == nil {
		return 0.0
	}

	score := s.bias
	score += s.weights["bid_price"] * math.Log1p(bid.Price)

	relevance := 0.3 + s.rng.Float64()*0.7
	score += s.weights["relevance"] * relevance

	ctrBase := 0.01 + s.rng.Float64()*0.05
	score += s.weights["ctr_base"] * ctrBase

	cvrBase := 0.005 + s.rng.Float64()*0.02
	score += s.weights["cvr_base"] * cvrBase

	freshness := s.rng.Float64()
	score += s.weights["freshness"] * freshness

	return score
}

func (s *LRScorer) ScoreWithECPM(bid *model.Bid) float64 {
	score := s.Score(bid)
	return score * bid.Price * 1000
}

func ScoreMultiple(scorer *LRScorer, bids []model.Bid) map[string]float64 {
	scores := make(map[string]float64, len(bids))
	for _, bid := range bids {
		scores[bid.ID] = scorer.ScoreWithECPM(&bid)
	}
	return scores
}
