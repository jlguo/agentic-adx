// Package ab implements the A/B experiment framework — a first-class component
// that deterministically routes traffic between control and treatment variants using
// CRC32-based hashing. All assignments are O(1) hash computations that never block
// the RTB hot path.
package ab

import (
	"database/sql"
	"fmt"
	"hash/crc32"
	"log"
	"sync"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

// ExperimentStatus represents the lifecycle state of an experiment.
type ExperimentStatus string

const (
	StatusActive    ExperimentStatus = "running"
	StatusPaused    ExperimentStatus = "paused"
	StatusCompleted ExperimentStatus = "completed"
)

// Experiment represents an A/B experiment configuration loaded from MySQL.
type Experiment struct {
	ID           int64
	Name         string
	TrafficRatio float64 // 0.0 – 1.0, fraction going to treatment
	HashSalt     string
	Status       ExperimentStatus
	StartTime    time.Time
}

// ExperimentAssignment holds the routing decision for a single request.
type ExperimentAssignment struct {
	ExperimentID int64
	Variant      string // "control" or "treatment"
}

// ExperimentManager holds active experiments and periodically refreshes them from MySQL.
// Reads are safe under sync.RWMutex. Supports first-match-wins for multiple experiments
// (each uses independent salts for traffic isolation).
type ExperimentManager struct {
	db          *sql.DB
	mu          sync.RWMutex
	experiments []*Experiment
	done        chan struct{}
}

// NewExperimentManager connects to MySQL, loads active experiments, and starts the
// 60-second refresh loop. Returns an error if the database connection fails.
func NewExperimentManager(mysqlDSN string) (*ExperimentManager, error) {
	db, err := sql.Open("mysql", mysqlDSN)
	if err != nil {
		return nil, fmt.Errorf("ab: mysql open: %w", err)
	}

	db.SetMaxOpenConns(5)
	db.SetMaxIdleConns(2)
	db.SetConnMaxLifetime(5 * time.Minute)

	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ab: mysql ping: %w", err)
	}

	mgr := &ExperimentManager{
		db:   db,
		done: make(chan struct{}),
	}

	if err := mgr.loadExperiments(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ab: initial load: %w", err)
	}

	go mgr.refreshLoop()

	log.Printf("AB manager initialized: %d active experiments", len(mgr.experiments))
	return mgr, nil
}

// loadExperiments queries active experiments from MySQL.
func (mgr *ExperimentManager) loadExperiments() error {
	query := `SELECT id, name, traffic_ratio, hash_salt, status, started_at
	           FROM experiments
	           WHERE status = 'running'
	           ORDER BY id`

	rows, err := mgr.db.Query(query)
	if err != nil {
		return fmt.Errorf("ab: query experiments: %w", err)
	}
	defer rows.Close()

	var experiments []*Experiment
	for rows.Next() {
		exp := &Experiment{}
		if err := rows.Scan(&exp.ID, &exp.Name, &exp.TrafficRatio,
			&exp.HashSalt, &exp.Status, &exp.StartTime); err != nil {
			return fmt.Errorf("ab: scan experiment: %w", err)
		}
		experiments = append(experiments, exp)
	}

	if err := rows.Err(); err != nil {
		return fmt.Errorf("ab: rows iteration: %w", err)
	}

	mgr.mu.Lock()
	mgr.experiments = experiments
	mgr.mu.Unlock()

	return nil
}

// AssignExperiment deterministically assigns a request to an experiment variant.
//
// For each active experiment (in order): hash = CRC32(salt + reqID) % 100.
//   - hash < traffic_ratio * 100 → treatment
//   - hash >= traffic_ratio * 100 → control
//
// First-match-wins: the first experiment whose bucket captures the request
// claims it. Different salts produce independent hash spaces, enabling
// traffic isolation across concurrent experiments.
//
// Returns nil when no active experiments exist.
func (mgr *ExperimentManager) AssignExperiment(reqID string) *ExperimentAssignment {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	for _, exp := range mgr.experiments {
		bucket := crc32.ChecksumIEEE([]byte(exp.HashSalt+reqID)) % 100
		threshold := int(exp.TrafficRatio * 100)

		if threshold > 0 && bucket < uint32(threshold) {
			return &ExperimentAssignment{
				ExperimentID: exp.ID,
				Variant:      "treatment",
			}
		}
		// Request falls into this experiment's control bucket.
		return &ExperimentAssignment{
			ExperimentID: exp.ID,
			Variant:      "control",
		}
	}

	return nil
}

// GetExperiment returns an experiment by ID, or nil if not found.
func (mgr *ExperimentManager) GetExperiment(id int64) *Experiment {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	for _, exp := range mgr.experiments {
		if exp.ID == id {
			return exp
		}
	}
	return nil
}

// refreshLoop reloads experiments from MySQL every 60 seconds.
func (mgr *ExperimentManager) refreshLoop() {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := mgr.loadExperiments(); err != nil {
				log.Printf("ab: refresh error: %v", err)
			}
		case <-mgr.done:
			return
		}
	}
}

// Close shuts down the experiment manager and releases the database connection.
func (mgr *ExperimentManager) Close() error {
	close(mgr.done)
	return mgr.db.Close()
}
