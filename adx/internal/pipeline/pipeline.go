package pipeline

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/agentic-adx/adx/internal/ab"
	"github.com/agentic-adx/adx/internal/auction"
	"github.com/agentic-adx/adx/internal/baseline"
	"github.com/agentic-adx/adx/internal/event"
	"github.com/agentic-adx/adx/internal/fallback"
	"github.com/agentic-adx/adx/internal/gpr"
	"github.com/agentic-adx/adx/internal/model"
	"github.com/agentic-adx/adx/internal/vector"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type Pipeline struct {
	qrClient       *vector.QdrantClient
	gpClient       *gpr.GPRClient
	lrScorer       *fallback.LRScorer
	baselineScorer *baseline.DeepFMScorer
	abManager      *ab.ExperimentManager
	eventProducer  *event.EventProducer
	scoreCache     *gpr.ScoreCache
	tracer         trace.Tracer
}

type PipelineResult struct {
	BidResponse     model.BidResponse
	AuctionResult   *auction.AuctionResult
	GPRUsed         bool
	ExperimentID    string
	Variant         string
	RecallCount     int
	RecallMs        float64
	GPRMs           float64
	AuctionMs       float64
	TotalMs         float64
}

func NewPipeline(qr *vector.QdrantClient, gp *gpr.GPRClient) *Pipeline {
	return &Pipeline{
		qrClient: qr,
		gpClient: gp,
		lrScorer: fallback.NewLRScorer(),
	}
}

func (p *Pipeline) SetEventProducer(ep *event.EventProducer) {
	p.eventProducer = ep
}

func (p *Pipeline) SetABManager(mgr *ab.ExperimentManager) {
	p.abManager = mgr
}

func (p *Pipeline) SetBaselineScorer(scorer *baseline.DeepFMScorer) {
	p.baselineScorer = scorer
}

func (p *Pipeline) SetTracer(t trace.Tracer) {
	p.tracer = t
}

func (p *Pipeline) SetScoreCache(sc *gpr.ScoreCache) {
	p.scoreCache = sc
}

func (p *Pipeline) Process(ctx context.Context, req *model.BidRequest) (*PipelineResult, error) {
	start := time.Now()
	result := &PipelineResult{}
	var totalStart = start

	tracer := p.tracer
	if tracer == nil {
		tracer = otel.Tracer("adx-core")
	}

	// Step 1: Vector recall
	var recallSpan trace.Span
	if tracer != nil {
		_, recallSpan = tracer.Start(ctx, "vector_recall")
		defer recallSpan.End()
	}
	recallStart := time.Now()
	userVector := extractUserVector(req)

	var candidates []vector.ScoredAd
	if p.qrClient != nil {
		var searchErr error
		candidates, searchErr = p.qrClient.Search(ctx, userVector, 100)
		if searchErr != nil {
			log.Printf("vector recall error: %v", searchErr)
		}
	}
	result.RecallMs = float64(time.Since(recallStart).Microseconds()) / 1000.0

	result.RecallCount = len(candidates)
	if recallSpan != nil {
		recallSpan.SetAttributes(attribute.Int("recall_count", len(candidates)))
	}

	// Step 2: A/B experiment routing (deterministic, O(1) hash computation).
	// This runs before scoring so the pipeline can choose GPR (treatment) or
	// baseline DeepFM (control) based on the experiment assignment.
	var abAssign *ab.ExperimentAssignment
	if p.abManager != nil {
		abAssign = p.abManager.AssignExperiment(req.ID)
		if abAssign != nil {
			result.ExperimentID = fmt.Sprintf("%d", abAssign.ExperimentID)
			result.Variant = abAssign.Variant
		}
	}

	gprAds := make([]gpr.GPRRequest, len(candidates))
	for i, c := range candidates {
		gprAds[i] = gpr.GPRRequest{
			AdID:       c.AdID,
			AdTitle:    c.Title,
			AdCategory: c.Category,
			BidPrice:   c.BidPrice,
		}
	}

	// Step 3: Scoring — GPR (treatment) or baseline DeepFM (control).
	gprStart := time.Now()
	userCtx := buildUserContext(req)

	var scores []gpr.GPRResponse
	var gprErr error

	useExperimentalScorer := false
	if abAssign != nil && abAssign.Variant == "control" && p.baselineScorer != nil {
		// Control variant: use baseline DeepFM scorer instead of GPR.
		if tracer != nil {
			_, baselineSpan := tracer.Start(ctx, "baseline_score")
			defer baselineSpan.End()
			baselineSpan.SetAttributes(attribute.Bool("treatment", false))
		}
		fmFeatures := make([]baseline.AdFeature, len(candidates))
		for i, c := range candidates {
			fmFeatures[i] = baseline.AdFeature{
				AdID:      c.AdID,
				Title:     c.Title,
				Category:  c.Category,
				BidPrice:  c.BidPrice,
			}
		}
		fmResults := p.baselineScorer.Score(userCtx, fmFeatures)
		// Convert baseline results to GPRResponse format for downstream compatibility.
		scores = make([]gpr.GPRResponse, len(fmResults))
		for i, r := range fmResults {
			scores[i] = gpr.GPRResponse{
				AdID: r.AdID,
				CTR:  r.CTR,
				CVR:  r.CVR,
				ECPM: r.ECPM,
			}
		}
		result.GPRUsed = false
		useExperimentalScorer = true
		log.Printf("AB: experiment %s variant=control — using baseline DeepFM scorer",
			result.ExperimentID)
	} else {
		if tracer != nil {
			_, gprSpan := tracer.Start(ctx, "gpr_score")
			defer gprSpan.End()
			gprSpan.SetAttributes(attribute.Bool("treatment", true))
		}

		// Check Redis score cache (pre-computed by cpu_scorer.py).
		cacheHit := false
		if p.scoreCache != nil {
			adIDs := make([]int64, len(candidates))
			for i, c := range candidates {
				adIDs[i] = c.AdID
			}
			cached, err := p.scoreCache.GetScores(ctx, adIDs)
			if err != nil {
				log.Printf("score cache read error: %v", err)
			} else if len(cached) == len(candidates) {
				// All candidates have cached scores — use them directly.
				scores = make([]gpr.GPRResponse, len(candidates))
				for i, c := range candidates {
					if s, ok := cached[c.AdID]; ok {
						scores[i] = s
					}
				}
				result.GPRUsed = true
				cacheHit = true
				log.Printf("score cache: used pre-computed scores for %d ads", len(candidates))
			} else if len(cached) > 0 {
				log.Printf("score cache: partial hit (%d/%d), falling back to GPR client",
					len(cached), len(candidates))
			}
		}

		if !cacheHit {
			scores, gprErr = p.gpClient.Score(ctx, userCtx, gprAds)
			if gprErr != nil {
				log.Printf("gpr scoring error (falling back to LR): %v", gprErr)
				result.GPRUsed = false
			} else {
				result.GPRUsed = true
			}
		}
	}
	result.GPRMs = float64(time.Since(gprStart).Microseconds()) / 1000.0
	_ = useExperimentalScorer

	// Step 4: Build bids with scores.
	bids := make([]model.Bid, len(candidates))
	for i, c := range candidates {
		bid := model.Bid{
			ID:    fmt.Sprintf("bid-%d", c.AdID),
			ImpID: req.Imp[0].ID,
			Price: c.BidPrice,
			AdID:  fmt.Sprintf("%d", c.AdID),
		}

		if result.GPRUsed && i < len(scores) {
			bid.Price = scores[i].ECPM / 1000.0
		}
		// When using baseline scorer (GPRUsed=false), also apply the scores.
		if !result.GPRUsed && i < len(scores) {
			bid.Price = scores[i].ECPM / 1000.0
		}
		bids[i] = bid
	}

	// Step 5: Auction.
	if tracer != nil {
		_, auctionSpan := tracer.Start(ctx, "auction")
		defer auctionSpan.End()
	}
	auctionStart := time.Now()
	ecpmScores := make(map[string]float64)
	if result.GPRUsed {
		for _, s := range scores {
			ecpmScores[fmt.Sprintf("bid-%d", s.AdID)] = s.ECPM
		}
	} else if len(scores) > 0 {
		for _, s := range scores {
			ecpmScores[fmt.Sprintf("bid-%d", s.AdID)] = s.ECPM
		}
	} else {
		for i, c := range candidates {
			ecpmScores[fmt.Sprintf("bid-%d", c.AdID)] = c.BidPrice * 0.02 * 1000
			_ = i
		}
	}

	auctionResult := auction.RunWithECPM(bids, ecpmScores, 0.10)
	result.AuctionResult = auctionResult
	result.AuctionMs = float64(time.Since(auctionStart).Microseconds()) / 1000.0

	// Build response
	bidResp := model.BidResponse{
		ID:    req.ID,
		BidID: req.ID + "-resp",
	}

	if !auctionResult.IsEmpty {
		bidResp.SeatBid = []model.SeatBid{
			{Bid: []model.Bid{*auctionResult.Winner}},
		}
	}

	result.BidResponse = bidResp
	result.TotalMs = float64(time.Since(totalStart).Microseconds()) / 1000.0

	// Fire-and-forget impression event when there is a winner (must NOT block the RTB hot path).
	if !auctionResult.IsEmpty && p.eventProducer != nil {
		go func() {
			p.eventProducer.PublishImpression(context.Background(),
				req.ID, auctionResult.Winner.AdID, auctionResult.Winner.Price,
				result.ExperimentID, result.Variant,
			)
		}()
	}

	return result, nil
}

func extractUserVector(req *model.BidRequest) []float32 {
	vec := make([]float32, 384)
	for i := range vec {
		vec[i] = float32(i%7) * 0.1
	}
	return vec
}

func buildUserContext(req *model.BidRequest) string {
	ctx := "User browsing"
	if req.Site != nil {
		ctx += fmt.Sprintf(" on %s", req.Site.Domain)
	}
	if req.Device != nil {
		ctx += fmt.Sprintf(" via %s", req.Device.UA[:min(30, len(req.Device.UA))])
	}
	if req.Device != nil && req.Device.Geo != nil {
		ctx += fmt.Sprintf(" from %s", req.Device.Geo.Country)
	}
	return ctx
}
