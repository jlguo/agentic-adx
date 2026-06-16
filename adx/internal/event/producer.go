package event

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/IBM/sarama"
)

// EventProducer wraps a Sarama async producer for fire-and-forget Kafka event publishing.
// All publish methods are non-blocking — they send on the async producer channel
// in a goroutine and return immediately. Must NOT block the RTB hot path.
type EventProducer struct {
	producer sarama.AsyncProducer
	broker   string
}

type adEvent struct {
	Timestamp    string  `json:"timestamp"`
	BidID        string  `json:"bid_id"`
	AdID         string  `json:"ad_id"`
	ECPM         float64 `json:"ecpm,omitempty"`
	Value        float64 `json:"value,omitempty"`
	ExperimentID string  `json:"experiment_id,omitempty"`
	Variant      string  `json:"variant,omitempty"`
}

// NewEventProducer creates a new Kafka event producer connected to the given broker.
func NewEventProducer(kafkaBroker string) (*EventProducer, error) {
	config := sarama.NewConfig()
	config.Producer.Return.Successes = false
	config.Producer.Return.Errors = false
	config.Producer.RequiredAcks = sarama.WaitForLocal
	config.Producer.Compression = sarama.CompressionSnappy
	config.Producer.Flush.Frequency = 100 * time.Millisecond
	config.Producer.Retry.Max = 3

	producer, err := sarama.NewAsyncProducer([]string{kafkaBroker}, config)
	if err != nil {
		return nil, fmt.Errorf("kafka producer init failed: %w", err)
	}

	// Drain async success/error channels in background so they don't block.
	go func() {
		for range producer.Successes() {
		}
	}()
	go func() {
		for err := range producer.Errors() {
			log.Printf("kafka producer error: %v", err)
		}
	}()

	log.Printf("Kafka producer connected: %s", kafkaBroker)
	return &EventProducer{producer: producer, broker: kafkaBroker}, nil
}

// PublishImpression publishes an impression event to the ad_impressions topic.
// Non-blocking; fires in a goroutine.
func (ep *EventProducer) PublishImpression(ctx context.Context, bidID, adID string, ecpm float64, experimentID, variant string) {
	event := adEvent{
		Timestamp:    time.Now().UTC().Format(time.RFC3339),
		BidID:        bidID,
		AdID:         adID,
		ECPM:         ecpm,
		ExperimentID: experimentID,
		Variant:      variant,
	}
	ep.publish("ad_impressions", event)
}

// PublishClick publishes a click event to the ad_clicks topic.
// Non-blocking; fires in a goroutine.
func (ep *EventProducer) PublishClick(ctx context.Context, bidID, adID string) {
	event := adEvent{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		BidID:     bidID,
		AdID:      adID,
	}
	ep.publish("ad_clicks", event)
}

// PublishConversion publishes a conversion event to the ad_conversions topic.
// Non-blocking; fires in a goroutine.
func (ep *EventProducer) PublishConversion(ctx context.Context, bidID, adID string, value float64) {
	event := adEvent{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		BidID:     bidID,
		AdID:      adID,
		Value:     value,
	}
	ep.publish("ad_conversions", event)
}

func (ep *EventProducer) publish(topic string, event adEvent) {
	data, err := json.Marshal(event)
	if err != nil {
		log.Printf("kafka marshal error (topic=%s): %v", topic, err)
		return
	}
	msg := &sarama.ProducerMessage{
		Topic: topic,
		Value: sarama.ByteEncoder(data),
	}
	ep.producer.Input() <- msg
}

// Close shuts down the Kafka producer gracefully.
func (ep *EventProducer) Close() error {
	return ep.producer.Close()
}
