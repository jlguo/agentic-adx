package auction

import (
	"testing"

	"github.com/agentic-adx/adx/internal/model"
)

func TestRun_EmptyBids(t *testing.T) {
	result := Run(nil, 0.10)
	if !result.IsEmpty {
		t.Error("expected empty auction with nil bids")
	}
}

func TestRun_FloorPrice(t *testing.T) {
	bids := []model.Bid{
		{ID: "bid-1", Price: 0.05},
		{ID: "bid-2", Price: 0.03},
	}
	result := Run(bids, 0.10)
	if !result.IsEmpty {
		t.Error("expected empty auction when all bids below floor")
	}
}

func TestRun_SingleBidder(t *testing.T) {
	bids := []model.Bid{
		{ID: "bid-1", Price: 2.50},
	}
	result := Run(bids, 0.10)
	if result.IsEmpty {
		t.Fatal("expected non-empty auction")
	}
	if result.Winner.ID != "bid-1" {
		t.Errorf("expected winner bid-1, got %s", result.Winner.ID)
	}
	if result.Price != 0.10 {
		t.Errorf("expected floor price 0.10 for single bidder, got %.2f", result.Price)
	}
}

func TestRun_SecondPriceAuction(t *testing.T) {
	bids := []model.Bid{
		{ID: "bid-1", Price: 3.00},
		{ID: "bid-2", Price: 2.00},
		{ID: "bid-3", Price: 1.00},
	}
	result := Run(bids, 0.10)
	if result.IsEmpty {
		t.Fatal("expected non-empty auction")
	}
	if result.Winner.ID != "bid-1" {
		t.Errorf("expected winner bid-1, got %s", result.Winner.ID)
	}
	if result.Price != 2.00 {
		t.Errorf("expected second price 2.00, got %.2f", result.Price)
	}
}

func TestRunWithECPM(t *testing.T) {
	bids := []model.Bid{
		{ID: "bid-1", Price: 1.00},
		{ID: "bid-2", Price: 3.00},
		{ID: "bid-3", Price: 2.00},
	}
	ecpm := map[string]float64{
		"bid-1": 0.50,
		"bid-2": 0.80,
		"bid-3": 0.60,
	}
	result := RunWithECPM(bids, ecpm, 0.10)
	if result.IsEmpty {
		t.Fatal("expected non-empty auction")
	}
	if result.Winner.ID != "bid-2" {
		t.Errorf("expected winner bid-2 (highest eCPM), got %s", result.Winner.ID)
	}
	if result.Price != 2.00 {
		t.Errorf("expected second price 2.00 (bid-3 price), got %.2f", result.Price)
	}
}

func TestRunWithECPM_FloorPrice(t *testing.T) {
	bids := []model.Bid{
		{ID: "bid-1", Price: 0.50},
		{ID: "bid-2", Price: 0.30},
	}
	ecpm := map[string]float64{
		"bid-1": 0.90,
		"bid-2": 0.80,
	}
	result := RunWithECPM(bids, ecpm, 1.00)
	if !result.IsEmpty {
		t.Error("expected empty auction when all bids below floor")
	}
}

func TestRunWithECPM_MissingECPM(t *testing.T) {
	bids := []model.Bid{
		{ID: "bid-1", Price: 2.00},
		{ID: "bid-2", Price: 1.00},
	}
	ecpm := map[string]float64{
		"bid-1": 0.80,
	}
	result := RunWithECPM(bids, ecpm, 0.10)
	if result.IsEmpty {
		t.Fatal("expected non-empty auction")
	}
	if result.Winner.ID != "bid-1" {
		t.Errorf("expected winner bid-1, got %s", result.Winner.ID)
	}
}

func TestRun_MixedBelowAndAboveFloor(t *testing.T) {
	bids := []model.Bid{
		{ID: "bid-low-1", Price: 0.01},
		{ID: "bid-ok", Price: 1.50},
		{ID: "bid-low-2", Price: 0.05},
	}
	result := Run(bids, 0.10)
	if result.IsEmpty {
		t.Fatal("expected non-empty auction")
	}
	if result.Winner.ID != "bid-ok" {
		t.Errorf("expected winner bid-ok, got %s", result.Winner.ID)
	}
	if result.Price != 0.10 {
		t.Errorf("expected floor price for single eligible bidder, got %.2f", result.Price)
	}
}
