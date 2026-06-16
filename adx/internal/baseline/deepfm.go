// Package baseline provides a traditional DeepFM-style scorer used as the control
// variant in A/B experiments. It approximates CTR/CVR/eCPM predictions using
// category priors and lightweight feature engineering — no GPU required.
package baseline

import (
	"math"
	"strings"
)

// AdFeature represents an ad candidate for scoring.
type AdFeature struct {
	AdID      int64
	Title     string
	Category  string
	Tags      []string
	BidPrice  float64
}

// ScoreResult holds the predicted metrics for a single ad.
type ScoreResult struct {
	AdID int64
	CTR  float64
	CVR  float64
	ECPM float64
}

// categoryStats holds pre-calibrated CTR and CVR baselines per category.
// These approximate real-world values derived from public datasets (Criteo/Avazu).
type categoryStats struct {
	CTRBase float64
	CVRBase float64
}

var categoryLookup = map[string]categoryStats{
	"electronics":    {CTRBase: 0.030, CVRBase: 0.020},
	"food_delivery":  {CTRBase: 0.040, CVRBase: 0.030},
	"fashion":        {CTRBase: 0.035, CVRBase: 0.025},
	"gaming":         {CTRBase: 0.050, CVRBase: 0.015},
	"travel":         {CTRBase: 0.025, CVRBase: 0.040},
	"general":        {CTRBase: 0.030, CVRBase: 0.020},
}

// defaultValue is used when no category match is found.
var defaultValue = categoryStats{CTRBase: 0.030, CVRBase: 0.020}

// DeepFMScorer implements traditional ad scoring using feature-engineered CTR/CVR
// predictions. Designed to match the GPR scoring interface for seamless A/B swapping.
type DeepFMScorer struct {
	weightCategory    float64
	weightPrice       float64
	weightTitleMatch  float64
}

// NewDeepFMScorer creates a scorer with sensible default feature weights.
func NewDeepFMScorer() *DeepFMScorer {
	return &DeepFMScorer{
		weightCategory:   0.50,
		weightPrice:      0.30,
		weightTitleMatch: 0.20,
	}
}

// Score computes CTR, CVR, and eCPM predictions for a batch of ad features.
// This mirrors the GPRClient.Score interface for easy swapping in the pipeline.
func (s *DeepFMScorer) Score(userContext string, adFeatures []AdFeature) []ScoreResult {
	results := make([]ScoreResult, len(adFeatures))
	for i, ad := range adFeatures {
		ctr := s.predictCTR(userContext, ad)
		cvr := s.predictCVR(ad.Category, ctr)
		ecpm := cvr * ad.BidPrice * 1000.0
		results[i] = ScoreResult{
			AdID: ad.AdID,
			CTR:  ctr,
			CVR:  cvr,
			ECPM: ecpm,
		}
	}
	return results
}

// predictCTR computes a sigmoid over the weighted sum of category, price, and text features.
//
//   CTR = sigmoid(
//     w_category * category_ctr_base +
//     w_price     * log1p(bid_price) +
//     w_title     * text_relevance(user_ctx, ad_title, ad_category)
//   )
func (s *DeepFMScorer) predictCTR(userContext string, ad AdFeature) float64 {
	stats := getCategoryStats(ad.Category)

	categoryScore := s.weightCategory * stats.CTRBase
	priceScore := s.weightPrice * math.Log1p(ad.BidPrice)
	textScore := s.weightTitleMatch * textRelevance(userContext, ad.Title, ad.Category)

	logit := categoryScore + priceScore + textScore
	return sigmoid(logit)
}

// predictCVR scales CTR by the category CVR baseline to produce a conversion estimate.
func (s *DeepFMScorer) predictCVR(category string, ctr float64) float64 {
	stats := getCategoryStats(category)
	return ctr * stats.CVRBase
}

// sigmoid implements the logistic function: 1 / (1 + e^(-x)).
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// getCategoryStats returns the calibration data for a category, falling back to
// the default on miss.
func getCategoryStats(category string) categoryStats {
	if stats, ok := categoryLookup[category]; ok {
		return stats
	}
	return defaultValue
}

// textRelevance computes a simple text match score between the user context and
// the ad's title / category. Returns a value in [0, 1].
//
// The approach: for each word in the user context, check if it appears as a
// substring of the ad title or category (case-insensitive). Count matches and
// normalize by the total number of context words.
func textRelevance(userContext, title, category string) float64 {
	if userContext == "" {
		return 0.3 // mild neutral prior when no user context
	}

	ctxLower := strings.ToLower(userContext)
	titleLower := strings.ToLower(title)
	categoryLower := strings.ToLower(category)

	words := strings.Fields(ctxLower)
	if len(words) == 0 {
		return 0.3
	}

	matches := 0
	for _, w := range words {
		if strings.Contains(titleLower, w) || strings.Contains(categoryLower, w) {
			matches++
		}
	}

	return float64(matches) / float64(len(words))
}
