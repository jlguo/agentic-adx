package baseline

import (
	"testing"
)

func TestDeepFMScorer_ReturnsAll(t *testing.T) {
	scorer := NewDeepFMScorer()
	ads := []AdFeature{
		{AdID: 1, Title: "Wireless Headphones", Category: "electronics", BidPrice: 5.0},
		{AdID: 2, Title: "Pizza Delivery", Category: "food_delivery", BidPrice: 3.0},
		{AdID: 3, Title: "RPG Mobile Game", Category: "gaming", BidPrice: 2.0},
	}

	results := scorer.Score("user browsing on example.com", ads)
	if len(results) != len(ads) {
		t.Fatalf("expected %d results, got %d", len(ads), len(results))
	}

	for i, r := range results {
		if r.AdID != ads[i].AdID {
			t.Errorf("result[%d].AdID = %d, want %d", i, r.AdID, ads[i].AdID)
		}
		if r.CTR <= 0 || r.CTR >= 1 {
			t.Errorf("result[%d].CTR = %f, expected in (0, 1)", i, r.CTR)
		}
		if r.CVR <= 0 || r.CVR >= 1 {
			t.Errorf("result[%d].CVR = %f, expected in (0, 1)", i, r.CVR)
		}
		if r.ECPM <= 0 {
			t.Errorf("result[%d].ECPM = %f, expected > 0", i, r.ECPM)
		}
	}
}

func TestDeepFMScorer_HigherBidHigherECPM(t *testing.T) {
	scorer := NewDeepFMScorer()
	ads := []AdFeature{
		{AdID: 1, Title: "Cheap Item", Category: "general", BidPrice: 1.0},
		{AdID: 2, Title: "Mid Item", Category: "general", BidPrice: 5.0},
		{AdID: 3, Title: "Expensive Item", Category: "general", BidPrice: 20.0},
	}

	results := scorer.Score("browsing", ads)
	if len(results) != 3 {
		t.Fatal("expected 3 results")
	}

	// Higher bid_price should yield higher eCPM when other factors are equal.
	if results[0].ECPM >= results[1].ECPM {
		t.Errorf("ecpm[%d]=%.4f >= ecpm[%d]=%.4f, higher bid should dominate",
			ads[0].AdID, results[0].ECPM, ads[1].AdID, results[1].ECPM)
	}
	if results[1].ECPM >= results[2].ECPM {
		t.Errorf("ecpm[%d]=%.4f >= ecpm[%d]=%.4f, higher bid should dominate",
			ads[1].AdID, results[1].ECPM, ads[2].AdID, results[2].ECPM)
	}
}

func TestDeepFMScorer_CategoryDifferences(t *testing.T) {
	scorer := NewDeepFMScorer()

	// Gaming has the highest CTR base (0.05), travel has the lowest (0.025).
	gaming := AdFeature{AdID: 1, Title: "RPG Game", Category: "gaming", BidPrice: 5.0}
	travel := AdFeature{AdID: 2, Title: "Flight Deal", Category: "travel", BidPrice: 5.0}

	results := scorer.Score("browsing", []AdFeature{gaming, travel})

	// Gaming should have higher CTR than travel at equal bid prices.
	if results[0].CTR <= results[1].CTR {
		t.Errorf("gaming CTR (%.4f) should be > travel CTR (%.4f)",
			results[0].CTR, results[1].CTR)
	}

	// Travel has higher CVR base (0.04 vs 0.015), so it may have higher CVR despite lower CTR.
	// CVR = CTR * cvr_base. Travel: ~0.025*sigmoid(...)*0.04 vs Gaming: ~0.05*sigmoid(...)*0.015
	// Gaming CVR ≈ 0.05*sigmoid*0.015 = 0.00075*sigmoid
	// Travel CVR ≈ 0.025*sigmoid*0.04 = 0.001*sigmoid
	// So travel should have higher CVR.
	if results[1].CVR <= results[0].CVR {
		t.Errorf("travel CVR (%.4f) should be > gaming CVR (%.4f)",
			results[1].CVR, results[0].CVR)
	}
}

func TestDeepFMScorer_EmptyContext(t *testing.T) {
	scorer := NewDeepFMScorer()
	ads := []AdFeature{
		{AdID: 1, Title: "Test Ad", Category: "general", BidPrice: 5.0},
	}

	results := scorer.Score("", ads)
	if results[0].CTR <= 0 {
		t.Error("CTR should be > 0 even with empty context")
	}
}

func TestTextRelevance(t *testing.T) {
	// Exact match.
	score := textRelevance("user likes electronics", "Electronics Sale", "electronics")
	if score <= 0 {
		t.Errorf("exact match relevance should be > 0, got %.3f", score)
	}

	// No match.
	score = textRelevance("user likes food", "Car Insurance", "insurance")
	if score >= 1.0 {
		t.Errorf("no-match relevance should be < 1, got %.3f", score)
	}

	// Empty context returns default.
	score = textRelevance("", "Anything", "general")
	if score != 0.3 {
		t.Errorf("empty context should return 0.3, got %.3f", score)
	}
}

func TestSigmoid(t *testing.T) {
	if sigmoid(0) != 0.5 {
		t.Errorf("sigmoid(0) = %f, want 0.5", sigmoid(0))
	}
	if sigmoid(100) < 0.9999 {
		t.Errorf("sigmoid(100) = %f, want ~1.0", sigmoid(100))
	}
	if sigmoid(-100) > 1e-40 {
		t.Errorf("sigmoid(-100) = %e, want ~0.0", sigmoid(-100))
	}
}

func TestGetCategoryStats_Fallback(t *testing.T) {
	stats := getCategoryStats("unknown_category")
	if stats.CTRBase != 0.030 || stats.CVRBase != 0.020 {
		t.Errorf("fallback stats wrong: %+v", stats)
	}
}
