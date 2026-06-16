package vector

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type QdrantClient struct {
	baseURL    string
	collection string
	http       *http.Client
}

type AdVector struct {
	AdID       int64
	CampaignID int64
	Title      string
	Category   string
	Tags       []string
	BidPrice   float64
	Vector     []float32
}

type qdrantSearchRequest struct {
	Vector      []float32 `json:"vector"`
	Limit       int       `json:"limit"`
	WithPayload bool      `json:"with_payload"`
}

type qdrantSearchResult struct {
	Result []struct {
		ID      uint64                 `json:"id"`
		Score   float32                `json:"score"`
		Payload map[string]interface{} `json:"payload"`
	} `json:"result"`
}

type qdrantPoint struct {
	ID      uint64                 `json:"id"`
	Vector  []float32              `json:"vector"`
	Payload map[string]interface{} `json:"payload"`
}

type qdrantUpsertRequest struct {
	Points []qdrantPoint `json:"points"`
}

type ScoredAd struct {
	AdID       int64
	CampaignID int64
	Title      string
	Category   string
	BidPrice   float64
	Score      float32
}

func NewQdrantClient(addr, collection string) (*QdrantClient, error) {
	client := &QdrantClient{
		baseURL:    fmt.Sprintf("http://%s", addr),
		collection: collection,
		http: &http.Client{
			Timeout: 1 * time.Second,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	if err := client.healthCheck(ctx); err != nil {
		return nil, err
	}

	return client, nil
}

func (c *QdrantClient) healthCheck(ctx context.Context) error {
	req, _ := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/healthz", nil)
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("qdrant health check: %w", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("qdrant unhealthy: %d", resp.StatusCode)
	}
	return nil
}

func (c *QdrantClient) Search(ctx context.Context, vector []float32, topK int) ([]ScoredAd, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Millisecond)
	defer cancel()

	body, _ := json.Marshal(qdrantSearchRequest{
		Vector:      vector,
		Limit:       topK,
		WithPayload: true,
	})

	url := fmt.Sprintf("%s/collections/%s/points/search", c.baseURL, c.collection)
	req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("qdrant search: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("qdrant search failed: %d %s", resp.StatusCode, string(bodyBytes))
	}

	var result qdrantSearchResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("qdrant decode: %w", err)
	}

	ads := make([]ScoredAd, 0, len(result.Result))
	for _, r := range result.Result {
		ad := ScoredAd{
			AdID:  int64(r.ID),
			Score: r.Score,
		}
		if p, ok := r.Payload["campaign_id"]; ok {
			if v, ok := p.(float64); ok {
				ad.CampaignID = int64(v)
			}
		}
		if p, ok := r.Payload["title"]; ok {
			ad.Title, _ = p.(string)
		}
		if p, ok := r.Payload["category"]; ok {
			ad.Category, _ = p.(string)
		}
		if p, ok := r.Payload["bid_price"]; ok {
			ad.BidPrice, _ = p.(float64)
		}
		ads = append(ads, ad)
	}

	return ads, nil
}

func (c *QdrantClient) Upsert(ctx context.Context, vectors []AdVector) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	points := make([]qdrantPoint, len(vectors))
	for i, v := range vectors {
		points[i] = qdrantPoint{
			ID:     uint64(v.AdID),
			Vector: v.Vector,
			Payload: map[string]interface{}{
				"campaign_id": v.CampaignID,
				"title":       v.Title,
				"category":    v.Category,
				"bid_price":   v.BidPrice,
			},
		}
	}

	body, _ := json.Marshal(qdrantUpsertRequest{Points: points})
	url := fmt.Sprintf("%s/collections/%s/points", c.baseURL, c.collection)

	req, _ := http.NewRequestWithContext(ctx, "PUT", url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("qdrant upsert: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant upsert failed: %d %s", resp.StatusCode, string(bodyBytes))
	}

	return nil
}
