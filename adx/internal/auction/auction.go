package auction

import (
	"sort"

	"github.com/agentic-adx/adx/internal/model"
)

type AuctionResult struct {
	Winner  *model.Bid
	Price   float64
	IsEmpty bool
}

func Run(bids []model.Bid, floorPrice float64) *AuctionResult {
	eligible := make([]model.Bid, 0, len(bids))
	for _, bid := range bids {
		if bid.Price >= floorPrice {
			eligible = append(eligible, bid)
		}
	}

	if len(eligible) == 0 {
		return &AuctionResult{IsEmpty: true}
	}

	sort.Slice(eligible, func(i, j int) bool {
		return eligible[i].Price > eligible[j].Price
	})

	winner := eligible[0]

	secondPrice := floorPrice
	if len(eligible) > 1 {
		secondPrice = eligible[1].Price
	}

	settlementPrice := secondPrice
	if settlementPrice < floorPrice {
		settlementPrice = floorPrice
	}

	return &AuctionResult{
		Winner:  &winner,
		Price:   settlementPrice,
		IsEmpty: false,
	}
}

func RunWithECPM(bids []model.Bid, ecpmScores map[string]float64, floorPrice float64) *AuctionResult {
	type scoredBid struct {
		bid   model.Bid
		ecpm  float64
	}

	scored := make([]scoredBid, 0, len(bids))
	for _, bid := range bids {
		ecpm, ok := ecpmScores[bid.ID]
		if !ok {
			continue
		}
		if bid.Price >= floorPrice {
			scored = append(scored, scoredBid{bid: bid, ecpm: ecpm})
		}
	}

	if len(scored) == 0 {
		return &AuctionResult{IsEmpty: true}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].ecpm > scored[j].ecpm
	})

	winner := scored[0].bid

	secondPrice := floorPrice
	if len(scored) > 1 {
		secondPrice = scored[1].bid.Price
	}

	settlementPrice := secondPrice
	if settlementPrice < floorPrice {
		settlementPrice = floorPrice
	}

	return &AuctionResult{
		Winner:  &winner,
		Price:   settlementPrice,
		IsEmpty: false,
	}
}
