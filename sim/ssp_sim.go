package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

var (
	targetURL  = flag.String("target", "http://localhost:8080/bid", "ADX bid endpoint URL")
	qps        = flag.Int("qps", 10, "Requests per second")
	duration   = flag.Int("duration", 0, "Run duration in seconds (0 = infinite)")
)

var userAgents = []string{
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
	"Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
	"Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/120.0.6099.144 Mobile Safari/537.36",
	"Mozilla/5.0 (Linux; Android 13; SM-S908B) AppleWebKit/537.36 Chrome/120.0.6099.144 Mobile Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
	"Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 Chrome/120.0.6099.144 Mobile Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edge/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.0 Safari/605.1.15",
	"Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 Version/16.0 Mobile/15E148 Safari/604.1",
	"Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
	"Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 Chrome/119.0.6045.163 Mobile Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36 OPR/105.0.0.0",
	"Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 Chrome/120.0.6099.144 Mobile Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.76",
	"Mozilla/5.0 (Linux; Android 11; Redmi Note 10) AppleWebKit/537.36 Chrome/119.0.6045.193 Mobile Safari/537.36",
}

var sites = []struct{ ID, Domain string }{
	{"site-news-001", "news.example.com"},
	{"site-sports-002", "sports.example.com"},
	{"site-tech-003", "tech.example.com"},
	{"site-lifestyle-004", "lifestyle.example.com"},
}

var countries = []string{"US", "CN", "GB", "JP", "DE", "FR", "KR", "BR", "IN", "AU"}

var bannerSizes = []struct{ W, H int }{
	{300, 250},
	{728, 90},
	{320, 50},
	{300, 600},
	{970, 250},
}

type stats struct {
	sent     atomic.Int64
	success  atomic.Int64
	fail     atomic.Int64
	latencies []int64
	mu       sync.Mutex
}

func (s *stats) record(d time.Duration) {
	s.mu.Lock()
	s.latencies = append(s.latencies, d.Milliseconds())
	s.mu.Unlock()
}

func (s *stats) snapshot() (avg, p50, p99 float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.latencies) == 0 {
		return 0, 0, 0
	}
	cp := make([]int64, len(s.latencies))
	copy(cp, s.latencies)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })

	var sum int64
	for _, v := range cp {
		sum += v
	}
	avg = float64(sum) / float64(len(cp))
	p50 = float64(cp[len(cp)/2])
	p99 = float64(cp[int(float64(len(cp))*0.99)])
	return
}

func generateBidRequest() map[string]interface{} {
	site := sites[rand.Intn(len(sites))]
	banner := bannerSizes[rand.Intn(len(bannerSizes))]
	country := countries[rand.Intn(len(countries))]

	userID := ""
	if rand.Float64() > 0.3 {
		userID = fmt.Sprintf("user-%08d", rand.Intn(100000))
	}

	return map[string]interface{}{
		"id": fmt.Sprintf("req-%d-%08d", time.Now().Unix(), rand.Intn(100000000)),
		"imp": []map[string]interface{}{
			{
				"id":       fmt.Sprintf("imp-%d", rand.Intn(1000)),
				"bidfloor": rand.Float64()*4.9 + 0.1,
				"banner": map[string]interface{}{
					"w": banner.W,
					"h": banner.H,
				},
			},
		},
		"site": map[string]interface{}{
			"id":     site.ID,
			"domain": site.Domain,
			"page":   fmt.Sprintf("https://%s/article/%d", site.Domain, rand.Intn(10000)),
		},
		"device": map[string]interface{}{
			"ua": userAgents[rand.Intn(len(userAgents))],
			"ip": fmt.Sprintf("%d.%d.%d.%d", rand.Intn(256), rand.Intn(256), rand.Intn(256), rand.Intn(256)),
			"geo": map[string]interface{}{
				"country": country,
			},
		},
		"user": map[string]interface{}{
			"id": userID,
		},
	}
}

func main() {
	flag.Parse()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	_ = rng

	st := &stats{latencies: make([]int64, 0, 10000)}
	client := &http.Client{
		Timeout: 5 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     30 * time.Second,
		},
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	var deadline time.Time
	if *duration > 0 {
		deadline = time.Now().Add(time.Duration(*duration) * time.Second)
	}

	interval := time.Second / time.Duration(*qps)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	reportTicker := time.NewTicker(1 * time.Second)
	defer reportTicker.Stop()

	log.Printf("SSP Simulator starting: target=%s qps=%d duration=%ds", *targetURL, *qps, *duration)

	for {
		select {
		case <-ctx.Done():
			printFinal(st)
			return
		case <-ticker.C:
			if *duration > 0 && time.Now().After(deadline) {
				printFinal(st)
				return
			}
			go sendBidRequest(client, st)
		case <-reportTicker.C:
			avg, p50, p99 := st.snapshot()
			log.Printf("sent=%d ok=%d fail=%d avg=%.1fms p50=%.1fms p99=%.1fms",
				st.sent.Load(), st.success.Load(), st.fail.Load(), avg, p50, p99)
		}
	}
}

func sendBidRequest(client *http.Client, st *stats) {
	reqBody := generateBidRequest()
	body, _ := json.Marshal(reqBody)

	start := time.Now()
	st.sent.Add(1)

	resp, err := client.Post(*targetURL, "application/json", bytes.NewReader(body))
	dur := time.Since(start)
	st.record(dur)

	if err != nil {
		st.fail.Add(1)
		log.Printf("REQ %s: error: %v", reqBody["id"], err)
		return
	}
	resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		st.success.Add(1)
	} else {
		st.fail.Add(1)
		log.Printf("REQ %s: HTTP %d (%.1fms)", reqBody["id"], resp.StatusCode, float64(dur.Microseconds())/1000.0)
	}
}

func printFinal(st *stats) {
	avg, p50, p99 := st.snapshot()
	log.Printf("=== Final Summary ===")
	log.Printf("Total sent:    %d", st.sent.Load())
	log.Printf("Success:       %d", st.success.Load())
	log.Printf("Failed:        %d", st.fail.Load())
	log.Printf("Avg latency:   %.1fms", avg)
	log.Printf("P50 latency:   %.1fms", p50)
	log.Printf("P99 latency:   %.1fms", p99)
	os.Exit(0)
}
