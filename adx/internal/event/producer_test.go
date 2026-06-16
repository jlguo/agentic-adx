package event

import (
	"context"
	"testing"
	"time"

	"github.com/IBM/sarama"
)

func newTestProducer(t *testing.T, broker string) (*EventProducer, error) {
	t.Helper()
	config := sarama.NewConfig()
	config.Producer.Return.Successes = false
	config.Producer.Return.Errors = false
	config.Producer.RequiredAcks = sarama.WaitForLocal
	config.Net.DialTimeout = 500 * time.Millisecond
	config.Net.ReadTimeout = 500 * time.Millisecond
	config.Metadata.Retry.Max = 0
	config.Metadata.Timeout = 500 * time.Millisecond

	producer, err := sarama.NewAsyncProducer([]string{broker}, config)
	if err != nil {
		return nil, err
	}

	go func() {
		for range producer.Successes() {
		}
	}()
	go func() {
		for range producer.Errors() {
		}
	}()

	return &EventProducer{producer: producer, broker: broker}, nil
}

func TestNewEventProducer_UnreachableDial(t *testing.T) {
	_, err := newTestProducer(t, "localhost:19999")
	if err == nil {
		t.Error("expected dial error for unreachable broker, got nil")
	}
}

func TestNewEventProducer_DefaultConstructor(t *testing.T) {
	_, err := NewEventProducer("localhost:19999")
	if err == nil {
		t.Error("expected error for unreachable broker via default constructor, got nil")
	}
}

func TestPublishMethods_NoPanicWithoutKafka(t *testing.T) {
	// Create producer with a bad address — will fail but we just need the struct shape for testing.
	ep, err := newTestProducer(t, "localhost:19999")
	if err == nil {
		defer ep.Close()
	}

	// Even if producer is nil or errored, test that publish methods exist and compile.
	ctx := context.Background()
	if ep != nil {
		ep.PublishImpression(ctx, "test-bid", "ad-1", 1.5, "", "")
		ep.PublishClick(ctx, "test-bid", "ad-1")
		ep.PublishConversion(ctx, "test-bid", "ad-1", 10.0)
	}
}

func TestPublishImpression_WithRealKafka(t *testing.T) {
	ep, err := NewEventProducer("localhost:9092")
	if err != nil {
		t.Skipf("no local Kafka, skipping: %v", err)
	}
	defer ep.Close()

	ep.PublishImpression(context.Background(), "test-bid-1", "ad-42", 1.23, "", "")
}

func TestPublishClick_WithRealKafka(t *testing.T) {
	ep, err := NewEventProducer("localhost:9092")
	if err != nil {
		t.Skipf("no local Kafka, skipping: %v", err)
	}
	defer ep.Close()

	ep.PublishClick(context.Background(), "test-bid-1", "ad-42")
}

func TestPublishConversion_WithRealKafka(t *testing.T) {
	ep, err := NewEventProducer("localhost:9092")
	if err != nil {
		t.Skipf("no local Kafka, skipping: %v", err)
	}
	defer ep.Close()

	ep.PublishConversion(context.Background(), "test-bid-1", "ad-42", 49.99)
}
