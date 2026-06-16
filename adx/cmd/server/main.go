package main

import (
	"context"
	"encoding/json"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/agentic-adx/adx/internal/ab"
	"github.com/agentic-adx/adx/internal/baseline"
	"github.com/agentic-adx/adx/internal/event"
	"github.com/agentic-adx/adx/internal/fallback"
	"github.com/agentic-adx/adx/internal/filter"
	"github.com/agentic-adx/adx/internal/gpr"
	"github.com/agentic-adx/adx/internal/model"
	"github.com/agentic-adx/adx/internal/pipeline"
	"github.com/agentic-adx/adx/internal/vector"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/sdk/resource"
	tracesdk "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
)

var (
	bidRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "adx_bid_requests_total",
			Help: "Total bid requests received",
		},
		[]string{"status"},
	)
	bidLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "adx_bid_latency_ms",
			Help:    "Bid processing latency in milliseconds",
			Buckets: prometheus.ExponentialBuckets(1, 1.5, 15),
		},
	)
)

var (
	p       *pipeline.Pipeline
	lr      *fallback.LRScorer
	rf      *filter.RedisFilter
	ep      *event.EventProducer
	abMgr   *ab.ExperimentManager
	tracer  trace.Tracer
)

func init() {
	prometheus.MustRegister(bidRequestsTotal)
	prometheus.MustRegister(bidLatency)
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	redisAddr := envOrDefault("REDIS_ADDR", "localhost:6379")
	qdrantAddr := envOrDefault("QDRANT_ADDR", "localhost:6333")
	gprAddr := envOrDefault("GPR_ADDR", "localhost:8000")

	// GPR_REDIS_ADDR allows the score cache to use a separate Redis instance
	// (e.g., on the GPU host where the scorer writes). Falls back to REDIS_ADDR.
	gprRedisAddr := envOrDefault("GPR_REDIS_ADDR", redisAddr)

	var err error
	rf, err = filter.NewRedisFilter(redisAddr)
	if err != nil {
		log.Printf("WARNING: Redis unavailable (%v) — filters disabled", err)
	} else {
		defer rf.Close()
		log.Printf("Redis connected: %s", redisAddr)
	}

	qrClient, err := vector.NewQdrantClient(qdrantAddr, "ad_vectors")
	if err != nil {
		log.Printf("WARNING: Qdrant unavailable (%v) — vector recall disabled", err)
	} else {
		log.Printf("Qdrant connected: %s", qdrantAddr)
	}

	gpClient := gpr.NewGPRClient(gprAddr)
	log.Printf("GPR client ready: %s", gprAddr)

	p = pipeline.NewPipeline(qrClient, gpClient)
	lr = fallback.NewLRScorer()

	tp, jaegerAddr, err := initTracer()
	if err != nil {
		log.Printf("WARNING: tracing unavailable (%v)", err)
	} else {
		defer func() {
			if err := tp.Shutdown(ctx); err != nil {
				log.Printf("tracer shutdown error: %v", err)
			}
		}()
		tracer = tp.Tracer("adx-core")
		p.SetTracer(tracer)
		log.Printf("Jaeger tracing connected: %s", jaegerAddr)
	}

	// Initialize A/B experiment manager (non-fatal if MySQL is unavailable).
	mysqlDSN := envOrDefault("MYSQL_DSN", "adx:adx_pass@tcp(localhost:3306)/adx?parseTime=true")
	abMgr, err = ab.NewExperimentManager(mysqlDSN)
	if err != nil {
		log.Printf("WARNING: AB manager unavailable (%v) — experiments disabled", err)
	} else {
		p.SetABManager(abMgr)
		p.SetBaselineScorer(baseline.NewDeepFMScorer())
		log.Printf("AB manager connected: %s", mysqlDSN)
	}

	if sc, err := gpr.NewScoreCache(gprRedisAddr); err != nil {
		log.Printf("WARNING: GPR score cache unavailable at %s (%v) — using GPR client directly",
			gprRedisAddr, err)
	} else {
		p.SetScoreCache(sc)
		log.Printf("GPR score cache connected: %s", gprRedisAddr)
	}

	kafkaAddr := envOrDefault("KAFKA_ADDR", "localhost:9092")
	ep, err = event.NewEventProducer(kafkaAddr)
	if err != nil {
		log.Printf("WARNING: Kafka unavailable (%v) — event publishing disabled", err)
	} else {
		p.SetEventProducer(ep)
		log.Printf("Kafka connected: %s", kafkaAddr)
	}

	router := gin.New()
	router.Use(gin.Recovery())
	router.GET("/health", handleHealth)
	router.POST("/bid", handleBid)
	router.GET("/api/creative/:id", handleCreative)

	go func() {
		srv := &http.Server{Addr: ":8080", Handler: router}
		log.Println("HTTP server listening on :8080")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	go func() {
		metricsMux := http.NewServeMux()
		metricsMux.Handle("/metrics", promhttp.Handler())
		metricsSrv := &http.Server{Addr: ":9091", Handler: metricsMux}
		log.Println("Metrics server listening on :9091")
		if err := metricsSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Metrics server error: %v", err)
		}
	}()

	go func() {
		lis, err := net.Listen("tcp", ":9090")
		if err != nil {
			log.Fatalf("gRPC listen error: %v", err)
		}
		grpcSrv := grpc.NewServer()
		log.Println("gRPC server listening on :9090")
		if err := grpcSrv.Serve(lis); err != nil {
			log.Fatalf("gRPC server error: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("Shutting down...")
}

func handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func handleCreative(c *gin.Context) {
	idStr := c.Param("id")
	if abMgr == nil || abMgr.DB() == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "database unavailable"})
		return
	}

	var cr model.Creative
	var tagsStr string
	err := abMgr.DB().QueryRowContext(c.Request.Context(),
		`SELECT id, campaign_id, COALESCE(title,''), COALESCE(description,''),
		        COALESCE(image_url,''), COALESCE(landing_url,''),
		        COALESCE(category,''), COALESCE(tags,'[]'), COALESCE(status,'')
		 FROM creatives WHERE id = ?`, idStr).
		Scan(&cr.ID, &cr.CampaignID, &cr.Title, &cr.Description,
			&cr.ImageURL, &cr.LandingURL, &cr.Category, &tagsStr,
			&cr.Status)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "creative not found"})
		return
	}
	cr.Tags = json.RawMessage(tagsStr)
	c.JSON(http.StatusOK, cr)
}

func handleBid(c *gin.Context) {
	tracer := otel.Tracer("adx-core")
	ctx, span := tracer.Start(c.Request.Context(), "bid_request")
	defer span.End()

	start := time.Now()
	defer func() {
		bidLatency.Observe(float64(time.Since(start).Milliseconds()))
	}()

	var req model.BidRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		bidRequestsTotal.WithLabelValues("parse_error").Inc()
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid bid request"})
		return
	}

	if err := req.Validate(); err != nil {
		bidRequestsTotal.WithLabelValues("validation_error").Inc()
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	span.SetAttributes(attribute.String("request_id", req.ID))

	if rf != nil {
		_ = filter.Apply(&req)
	}

	if p == nil {
		bidRequestsTotal.WithLabelValues("success_stub").Inc()
		c.JSON(http.StatusOK, model.BidResponse{
			ID:    req.ID,
			BidID: req.ID + "-bid",
		})
		return
	}

	result, err := p.Process(ctx, &req)
	if err != nil {
		log.Printf("pipeline error: %v", err)
		bidRequestsTotal.WithLabelValues("error").Inc()
		c.JSON(http.StatusOK, model.BidResponse{
			ID:    req.ID,
			BidID: req.ID + "-bid",
		})
		return
	}

	gprLabel := "lr"
	if result.GPRUsed {
		gprLabel = "gpr"
	}
	bidRequestsTotal.WithLabelValues("success_" + gprLabel).Inc()
	log.Printf("bid %s: recall=%d in %.1fms, gpr=%v in %.1fms, auction in %.1fms, total=%.1fms",
		req.ID, result.RecallCount, result.RecallMs, result.GPRUsed, result.GPRMs,
		result.AuctionMs, result.TotalMs)

	span.SetAttributes(
		attribute.Bool("gpr_used", result.GPRUsed),
		attribute.Float64("latency_ms", result.TotalMs),
	)

	c.JSON(http.StatusOK, result.BidResponse)
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func initTracer() (*tracesdk.TracerProvider, string, error) {
	jaegerAddr := envOrDefault("JAEGER_ADDR", "http://localhost:14268/api/traces")
	exp, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(jaegerAddr)))
	if err != nil {
		return nil, "", err
	}
	tp := tracesdk.NewTracerProvider(
		tracesdk.WithBatcher(exp),
		tracesdk.WithResource(resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceName("adx-core"),
		)),
		tracesdk.WithSampler(tracesdk.AlwaysSample()),
	)
	otel.SetTracerProvider(tp)
	return tp, jaegerAddr, nil
}
