package rebalancer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// ---------- Counter ----------

// Counter tracks EWMA routing frequency per (layer, expert).
// Decay is time-based: ema *= exp(-dt/tau), decoupled from token rate.
type Counter struct {
	mu       sync.Mutex
	nLayers  int
	nExperts int
	ema      [][]float64
	lastTime time.Time
	tau      time.Duration
}

func NewCounter(nLayers, nExperts int, tau time.Duration) *Counter {
	ema := make([][]float64, nLayers)
	for i := range ema {
		ema[i] = make([]float64, nExperts)
	}
	return &Counter{
		nLayers:  nLayers,
		nExperts: nExperts,
		ema:      ema,
		tau:      tau,
	}
}

func (c *Counter) Update(layer int, experts []int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.decay()

	for _, e := range experts {
		if layer >= 0 && layer < c.nLayers && e >= 0 && e < c.nExperts {
			c.ema[layer][e] += 1.0
		}
	}
}

// decay applies time-based exponential decay since last call.
func (c *Counter) decay() {
	now := time.Now()
	if c.lastTime.IsZero() {
		c.lastTime = now
		return
	}
	dt := now.Sub(c.lastTime).Seconds()
	c.lastTime = now
	if dt <= 0 || c.tau <= 0 {
		return
	}
	factor := 1.0 - dt/c.tau.Seconds()
	if factor <= 0 {
		for i := range c.ema {
			for j := range c.ema[i] {
				c.ema[i][j] = 0
			}
		}
		return
	}
	for i := range c.ema {
		for j := range c.ema[i] {
			c.ema[i][j] *= factor
		}
	}
}

type ExpertScore struct {
	ID    int
	Score float64
}

// Rank returns experts for a layer sorted by EWMA score, descending.
func (c *Counter) Rank(layer int) []ExpertScore {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.decay()

	if layer < 0 || layer >= c.nLayers {
		return nil
	}
	scores := make([]ExpertScore, c.nExperts)
	for i := 0; i < c.nExperts; i++ {
		scores[i] = ExpertScore{ID: i, Score: c.ema[layer][i]}
	}
	sort.SliceStable(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})
	return scores
}

// Snapshot returns a copy of all EWMA values for logging/debugging.
func (c *Counter) Snapshot() [][]float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.decay()

	out := make([][]float64, c.nLayers)
	for i := range c.ema {
		out[i] = make([]float64, c.nExperts)
		copy(out[i], c.ema[i])
	}
	return out
}

// ---------- Placement types ----------

type Change struct {
	Layer      int   `json:"layer"`
	ExpertsGPU []int `json:"experts_gpu"`
	Device     int   `json:"device"`
}

type Request struct {
	Changes []Change `json:"changes"`
}

type LayerInfo struct {
	Layer      int    `json:"layer"`
	OnGPU      bool   `json:"on_gpu"`
	Dynamic    bool   `json:"dynamic,omitempty"`
	ExpertsGPU []int  `json:"experts_gpu,omitempty"`
}

type GetResponse struct {
	Granularity string     `json:"granularity"`
	Layers      []LayerInfo `json:"layers"`
}

type ChangeResult struct {
	Layer      int   `json:"layer"`
	ExpertsGPU []int `json:"experts_gpu"`
	Device     int   `json:"device"`
	OK         bool  `json:"ok"`
}

type PostResponse struct {
	Changes []ChangeResult `json:"changes"`
}

// ---------- Placement client ----------

type Client struct {
	baseURL string
	http    *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *Client) Get() (*GetResponse, error) {
	resp, err := c.http.Get(c.baseURL + "/v1/model/expert-placement")
	if err != nil {
		return nil, fmt.Errorf("GET expert-placement: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("GET expert-placement: status %d: %s", resp.StatusCode, body)
	}

	var result GetResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode GET response: %w", err)
	}
	return &result, nil
}

func (c *Client) Post(changes []Change) (*PostResponse, error) {
	if len(changes) == 0 {
		return &PostResponse{}, nil
	}

	reqBody := Request{Changes: changes}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	resp, err := c.http.Post(
		c.baseURL+"/v1/model/expert-placement",
		"application/json",
		bytes.NewReader(bodyBytes),
	)
	if err != nil {
		return nil, fmt.Errorf("POST expert-placement: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("POST expert-placement: status %d: %s", resp.StatusCode, body)
	}

	var result PostResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode POST response: %w", err)
	}
	return &result, nil
}

// ---------- Rebalancer ----------

// Rebalancer computes target GPU expert sets from EWMA rankings,
// applying hysteresis to prevent thrashing at the K boundary.
type Rebalancer struct {
	nLayers    int
	topK       int
	hysteresis int
}

func New(nLayers, topK, hysteresis int) *Rebalancer {
	return &Rebalancer{
		nLayers:    nLayers,
		topK:       topK,
		hysteresis: hysteresis,
	}
}

// placementSet converts a slice of expert IDs to a set for quick lookup.
func placementSet(ids []int) map[int]bool {
	s := make(map[int]bool, len(ids))
	for _, id := range ids {
		s[id] = true
	}
	return s
}

// ComputeTarget determines the desired GPU expert set for a single layer.
//
// Hysteresis logic:
// - Start with current GPU set.
// - An expert stays placed if it's within top (K + hysteresis) by EWMA rank.
// - An expert gets promoted if it's within top (K - hysteresis) and there's room.
// - Fill remaining slots up to K from the ranking (promoting highest-ranked non-placed).
func (r *Rebalancer) ComputeTarget(layer int, ranked []ExpertScore, currentGPU []int) []int {
	if layer < 0 || layer >= r.nLayers {
		return nil
	}

	current := placementSet(currentGPU)
	target := make(map[int]bool)

	// Keep currently placed experts that are still within K+hysteresis rank.
	keepThreshold := r.topK + r.hysteresis
	if keepThreshold > len(ranked) {
		keepThreshold = len(ranked)
	}
	for i := 0; i < keepThreshold; i++ {
		if current[ranked[i].ID] {
			target[ranked[i].ID] = true
		}
	}

	// Promote from the ranking until we have K experts.
	for i := 0; i < len(ranked) && len(target) < r.topK; i++ {
		target[ranked[i].ID] = true
	}

	// Convert to sorted slice for stable comparison.
	result := make([]int, 0, len(target))
	for id := range target {
		result = append(result, id)
	}
	sort.Ints(result)
	return result
}

// Diff returns only the changes that differ from current placement.
// Returns nil if nothing changed.
func Diff(nLayers int, current []LayerInfo, target map[int][]int) []Change {
	changes := make([]Change, 0, nLayers)

	for layer := 0; layer < nLayers; layer++ {
		targetSet, hasTarget := target[layer]

		// Find current placement for this layer
		var currentGPU []int
		found := false
		for _, l := range current {
			if l.Layer == layer {
				if l.Dynamic {
					currentGPU = l.ExpertsGPU
				}
				found = true
				break
			}
		}

		if hasTarget {
			if !found || !intsEqual(currentGPU, targetSet) {
				changes = append(changes, Change{
					Layer:      layer,
					ExpertsGPU: targetSet,
					Device:     0,
				})
			}
		}
	}

	if len(changes) == 0 {
		return nil
	}
	return changes
}

func intsEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
