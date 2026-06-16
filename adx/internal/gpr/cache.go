package gpr

import (
	"context"
	"fmt"
	"strconv"

	"github.com/redis/go-redis/v9"
)

// ScoreCache reads pre-computed GPR scores from Redis.
// The cpu_scorer.py service writes HSET gpr_score:<ad_id> ctr X cvr Y ecpm Z.
type ScoreCache struct {
	client *redis.Client
}

// NewScoreCache creates a Redis-backed GPR score cache reader.
func NewScoreCache(addr string) (*ScoreCache, error) {
	client := redis.NewClient(&redis.Options{
		Addr: addr,
	})
	if err := client.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("score cache redis: %w", err)
	}
	return &ScoreCache{client: client}, nil
}

// GetScores reads cached GPR scores for the given ad IDs from Redis.
// Returns a map of ad_id → GPRResponse. Missing entries are simply absent.
func (c *ScoreCache) GetScores(ctx context.Context, adIDs []int64) (map[int64]GPRResponse, error) {
	if len(adIDs) == 0 {
		return nil, nil
	}

	pipe := c.client.Pipeline()
	cmds := make([]*redis.MapStringStringCmd, len(adIDs))
	for i, id := range adIDs {
		key := fmt.Sprintf("gpr_score:%d", id)
		cmds[i] = pipe.HGetAll(ctx, key)
	}
	if _, err := pipe.Exec(ctx); err != nil && err != redis.Nil {
		return nil, fmt.Errorf("score cache pipeline: %w", err)
	}

	scores := make(map[int64]GPRResponse, len(adIDs))
	for i, cmd := range cmds {
		data, err := cmd.Result()
		if err != nil || len(data) == 0 {
			continue // missing or empty → skip
		}
		ctr, _ := strconv.ParseFloat(data["ctr"], 64)
		cvr, _ := strconv.ParseFloat(data["cvr"], 64)
		ecpm, _ := strconv.ParseFloat(data["ecpm"], 64)
		scores[adIDs[i]] = GPRResponse{
			AdID: adIDs[i],
			CTR:  ctr,
			CVR:  cvr,
			ECPM: ecpm,
		}
	}
	return scores, nil
}

// Close releases the Redis connection.
func (c *ScoreCache) Close() error {
	return c.client.Close()
}
