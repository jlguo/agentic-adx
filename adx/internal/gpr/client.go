package gpr

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type GPRClient struct {
	baseURL string
	http    *http.Client
}

type GPRRequest struct {
	UserContext string  `json:"user_context"`
	AdID        int64   `json:"ad_id"`
	AdTitle     string  `json:"ad_title"`
	AdCategory  string  `json:"ad_category"`
	BidPrice    float64 `json:"bid_price"`
}

type GPRResponse struct {
	AdID   int64   `json:"ad_id"`
	CTR    float64 `json:"ctr"`
	CVR    float64 `json:"cvr"`
	ECPM   float64 `json:"ecpm"`
	Error  string  `json:"error,omitempty"`
}

type vLLMRequest struct {
	Messages    []vLLMMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream"`
}

type vLLMMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type vLLMResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func NewGPRClient(addr string) *GPRClient {
	return &GPRClient{
		baseURL: fmt.Sprintf("http://%s", addr),
		http: &http.Client{
			Timeout: 80 * time.Millisecond,
		},
	}
}

func (c *GPRClient) Score(ctx context.Context, userContext string, ads []GPRRequest) ([]GPRResponse, error) {
	if len(ads) == 0 {
		return nil, nil
	}

	prompt := buildScoringPrompt(userContext, ads)

	vllmReq := vLLMRequest{
		Messages: []vLLMMessage{
			{Role: "system", Content: "You are an ad scoring model. Output JSON only."},
			{Role: "user", Content: prompt},
		},
		MaxTokens:   256,
		Temperature: 0.0,
		Stream:      false,
	}

	body, _ := json.Marshal(vllmReq)
	req, _ := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return c.fallbackScores(ads), fmt.Errorf("gpr request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return c.fallbackScores(ads), fmt.Errorf("gpr status %d", resp.StatusCode)
	}

	respBody, _ := io.ReadAll(resp.Body)

	var vllmResp vLLMResponse
	if err := json.Unmarshal(respBody, &vllmResp); err != nil {
		return c.fallbackScores(ads), fmt.Errorf("gpr decode: %w", err)
	}

	if len(vllmResp.Choices) == 0 {
		return c.fallbackScores(ads), fmt.Errorf("gpr empty response")
	}

	var scores []GPRResponse
	if err := json.Unmarshal([]byte(vllmResp.Choices[0].Message.Content), &scores); err != nil {
		return c.fallbackScores(ads), fmt.Errorf("gpr parse scores: %w", err)
	}

	return scores, nil
}

func (c *GPRClient) fallbackScores(ads []GPRRequest) []GPRResponse {
	scores := make([]GPRResponse, len(ads))
	for i, ad := range ads {
		scores[i] = GPRResponse{
			AdID: ad.AdID,
			CTR:  0.02,
			CVR:  0.01,
			ECPM: ad.BidPrice * 0.02 * 1000,
		}
	}
	return scores
}

func buildScoringPrompt(userCtx string, ads []GPRRequest) string {
	buf := bytes.Buffer{}
	buf.WriteString("Score the following ads for user context: ")
	buf.WriteString(userCtx)
	buf.WriteString("\n\nAds to score:\n")
	for i, ad := range ads {
		fmt.Fprintf(&buf, "%d. [id=%d] %s (category: %s, bid: $%.2f)\n",
			i+1, ad.AdID, ad.AdTitle, ad.AdCategory, ad.BidPrice)
	}
	buf.WriteString("\nReturn JSON array with fields: ad_id, ctr, cvr, ecpm")
	return buf.String()
}
