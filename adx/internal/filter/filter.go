package filter

import (
	"context"
	"fmt"
	"time"

	"github.com/agentic-adx/adx/internal/model"
	"github.com/redis/go-redis/v9"
)

type RedisFilter struct {
	client *redis.Client
}

func NewRedisFilter(addr string) (*RedisFilter, error) {
	client := redis.NewClient(&redis.Options{
		Addr:         addr,
		DialTimeout:  50 * time.Millisecond,
		ReadTimeout:  50 * time.Millisecond,
		WriteTimeout: 50 * time.Millisecond,
		PoolSize:     100,
		MinIdleConns: 10,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis connection failed: %w", err)
	}

	return &RedisFilter{client: client}, nil
}

func (f *RedisFilter) CheckBudget(ctx context.Context, campaignID string) (bool, error) {
	key := fmt.Sprintf("budget:%s", campaignID)
	remaining, err := f.client.Get(ctx, key).Int64()
	if err == redis.Nil {
		return true, nil
	}
	if err != nil {
		return false, fmt.Errorf("budget check: %w", err)
	}
	return remaining > 0, nil
}

func (f *RedisFilter) DeductBudget(ctx context.Context, campaignID string, amount float64) error {
	key := fmt.Sprintf("budget:%s", campaignID)
	return f.client.DecrBy(ctx, key, int64(amount*100)).Err()
}

func (f *RedisFilter) CheckFrequency(ctx context.Context, userID, campaignID string, limit int, window time.Duration) (bool, error) {
	if userID == "" {
		return true, nil
	}
	key := fmt.Sprintf("freq:%s:%s", userID, campaignID)
	count, err := f.client.Incr(ctx, key).Result()
	if err != nil {
		return false, fmt.Errorf("frequency check: %w", err)
	}
	if count == 1 {
		f.client.Expire(ctx, key, window)
	}
	return int(count) <= limit, nil
}

func (f *RedisFilter) IsBlacklisted(ctx context.Context, keyType, value string) (bool, error) {
	exists, err := f.client.SIsMember(ctx, "blacklist:"+keyType, value).Result()
	if err != nil {
		return false, fmt.Errorf("blacklist check: %w", err)
	}
	return exists, nil
}

func (f *RedisFilter) Close() error {
	return f.client.Close()
}

func Apply(req *model.BidRequest) error {
	return nil
}

var (
	ErrBudgetExceeded    = fmt.Errorf("budget exceeded")
	ErrFrequencyExceeded = fmt.Errorf("frequency cap exceeded")
	ErrBlacklisted       = fmt.Errorf("blacklisted")
)
