package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestHealthEndpoint(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.GET("/health", handleHealth)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var body map[string]string
	json.Unmarshal(w.Body.Bytes(), &body)
	if body["status"] != "ok" {
		t.Errorf("expected status ok, got %s", body["status"])
	}
}

func TestBidEndpoint_ValidRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/bid", handleBid)

	bidReq := map[string]interface{}{
		"id": "test-req-001",
		"imp": []map[string]interface{}{
			{
				"id":       "imp-1",
				"bidfloor": 0.50,
				"banner": map[string]interface{}{
					"w": 300,
					"h": 250,
				},
			},
		},
		"site": map[string]interface{}{
			"id":     "site-001",
			"domain": "example.com",
		},
		"device": map[string]interface{}{
			"ua": "Mozilla/5.0",
			"ip": "192.168.1.1",
		},
	}

	body, _ := json.Marshal(bidReq)
	req := httptest.NewRequest(http.MethodPost, "/bid", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["id"] != "test-req-001" {
		t.Errorf("expected id test-req-001, got %v", resp["id"])
	}
}

func TestBidEndpoint_EmptyID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/bid", handleBid)

	bidReq := map[string]interface{}{
		"id": "",
		"imp": []map[string]interface{}{
			{
				"id": "imp-1",
			},
		},
	}

	body, _ := json.Marshal(bidReq)
	req := httptest.NewRequest(http.MethodPost, "/bid", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty id, got %d", w.Code)
	}
}

func TestBidEndpoint_InvalidJSON(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/bid", handleBid)

	req := httptest.NewRequest(http.MethodPost, "/bid", bytes.NewReader([]byte("not json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for invalid JSON, got %d", w.Code)
	}
}
