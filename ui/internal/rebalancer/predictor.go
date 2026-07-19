package rebalancer

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

type sseEvent struct {
	Layer   int   `json:"layer"`
	Experts []int `json:"experts"`
}

// Predictor owns the SSE consumer and the event-driven rebalance loop as a
// single self-contained component. Call Run(ctx) in a goroutine; it blocks
// until ctx is cancelled.
type Predictor struct {
	counter   *Counter
	rebal     *Rebalancer
	placement *Client
	endpoint  string
	interval  time.Duration
	dryRun    bool
	events    chan struct{}
}

func NewPredictor(backendURL string, nLayers, nExperts, topK, hysteresis int, tau, interval time.Duration, dryRun bool) *Predictor {
	return &Predictor{
		counter:   NewCounter(nLayers, nExperts, tau),
		rebal:     New(nLayers, topK, hysteresis),
		placement: NewClient(backendURL),
		endpoint:  strings.TrimRight(backendURL, "/") + "/v1/moe/routed-experts",
		interval:  interval,
		dryRun:    dryRun,
		events:    make(chan struct{}, 1),
	}
}

// Run starts the SSE consumer and the rebalance loop. Blocks until ctx is
// cancelled.
func (p *Predictor) Run(ctx context.Context) {
	go p.consumeSSELoop(ctx)
	p.runRebalanceLoop(ctx)
}

func (p *Predictor) consumeSSELoop(ctx context.Context) {
	const maxBackoff = 30 * time.Second
	backoff := time.Second

	for {
		if err := p.consumeSSE(ctx); err != nil {
			log.Printf("sse consumer: %v", err)
		}
		select {
		case <-ctx.Done():
			log.Printf("sse consumer shutting down")
			return
		case <-time.After(backoff):
		}
		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

func (p *Predictor) consumeSSE(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.endpoint, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Accept", "text/event-stream")

	client := &http.Client{Timeout: 0}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("connect: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		data := bytes.TrimPrefix(line, []byte("data: "))
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		var ev sseEvent
		if json.Unmarshal(data, &ev) == nil {
			p.counter.Update(ev.Layer, ev.Experts)
			select {
			case p.events <- struct{}{}:
			default:
			}
		}
	}
	return fmt.Errorf("stream closed")
}

// runRebalanceLoop drives rebalances from SSE events. The first event after
// idle triggers an immediate rebalance; subsequent events during the throttle
// window are coalesced into one rebalance when the window expires. If no
// events arrive the goroutine blocks indefinitely.
func (p *Predictor) runRebalanceLoop(ctx context.Context) {
	var (
		timer   *time.Timer
		timerC  <-chan time.Time
		pending bool
	)
	for {
		select {
		case <-p.events:
			if timer == nil {
				p.rebalance()
				timer = time.NewTimer(p.interval)
				timerC = timer.C
			} else {
				pending = true
			}
		case <-timerC:
			if pending {
				p.rebalance()
				pending = false
				timer.Reset(p.interval)
			} else {
				timer.Stop()
				timer = nil
				timerC = nil
			}
		case <-ctx.Done():
			if timer != nil {
				timer.Stop()
			}
			log.Printf("predictor shutting down")
			return
		}
	}
}

// rebalance fetches the current expert placement, computes target GPU expert
// sets from EWMA rankings, and POSTs the diff back to the backend.
func (p *Predictor) rebalance() {
	current, err := p.placement.Get()
	if err != nil {
		log.Printf("GET expert-placement failed: %v", err)
		return
	}

	nLayers := p.rebal.nLayers

	targets := make(map[int][]int)
	for layer := 0; layer < nLayers; layer++ {
		ranked := p.counter.Rank(layer)

		var currentGPU []int
		for _, l := range current.Layers {
			if l.Layer == layer && l.Dynamic {
				currentGPU = l.ExpertsGPU
				break
			}
		}

		target := p.rebal.ComputeTarget(layer, ranked, currentGPU)
		if len(target) > 0 {
			targets[layer] = target
		}
	}

	changes := Diff(nLayers, current.Layers, targets)
	if changes == nil {
		log.Printf("rebalance: no changes needed")
		return
	}

	currentMap := make(map[int][]int)
	for _, l := range current.Layers {
		if l.Dynamic {
			currentMap[l.Layer] = l.ExpertsGPU
		}
	}

	for _, c := range changes {
		prev := currentMap[c.Layer]
		promoted, demoted := diffExperts(prev, c.ExpertsGPU)
		log.Printf("rebalance layer %d: +%v -%v", c.Layer, promoted, demoted)
	}

	if p.dryRun {
		return
	}

	resp, err := p.placement.Post(changes)
	if err != nil {
		log.Printf("POST expert-placement failed: %v", err)
		return
	}

	for _, c := range resp.Changes {
		if !c.OK {
			log.Printf("rebalance layer %d: FAILED", c.Layer)
		}
	}
}

// diffExperts returns (promoted, demoted) experts between the previous and new
// GPU sets.
func diffExperts(prev, next []int) ([]int, []int) {
	prevSet := make(map[int]bool, len(prev))
	for _, e := range prev {
		prevSet[e] = true
	}
	nextSet := make(map[int]bool, len(next))
	for _, e := range next {
		nextSet[e] = true
	}

	var promoted, demoted []int
	for _, e := range next {
		if !prevSet[e] {
			promoted = append(promoted, e)
		}
	}
	for _, e := range prev {
		if !nextSet[e] {
			demoted = append(demoted, e)
		}
	}
	return promoted, demoted
}
