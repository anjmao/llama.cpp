package main

import (
	"bufio"
	"embed"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"time"
)

//go:embed index.html
var staticFS embed.FS

var (
	listenAddr = flag.String("listen", ":8081", "address to listen on")
	backendURL = flag.String("backend", "http://localhost:8080", "llama-server base URL")
)

func main() {
	flag.Parse()

	indexHTML, err := staticFS.ReadFile("index.html")
	if err != nil {
		log.Fatalf("failed to read index.html: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	mux.HandleFunc("/v1/moe/routed-experts", handleProxy)
	mux.HandleFunc("/v1/chat/completions", handleProxy)

	server := &http.Server{
		Addr:              *listenAddr,
		Handler:           mux,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      0,
		IdleTimeout:       120 * time.Second,
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("ui server listening on %s", *listenAddr)
	log.Printf("proxying /v1/* to %s", *backendURL)
	log.Fatal(server.ListenAndServe())
}

func handleProxy(w http.ResponseWriter, r *http.Request) {
	backend, err := url.Parse(*backendURL)
	if err != nil {
		http.Error(w, "invalid backend URL", http.StatusInternalServerError)
		return
	}

	target := backend.String() + r.URL.Path
	if r.URL.RawQuery != "" {
		target += "?" + r.URL.RawQuery
	}

	fmt.Printf("Proxy to %s\n", target)
	defer func() {
		fmt.Printf("Proxy to %s closed\n", target)
	}()

	req, err := http.NewRequestWithContext(r.Context(), r.Method, target, r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	for k, vv := range r.Header {
		for _, v := range vv {
			req.Header.Add(k, v)
		}
	}

	client := &http.Client{Timeout: 0}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	for k, vv := range resp.Header {
		for _, v := range vv {
			// let the backend set the content type; avoid length since we stream
			if k == "Content-Length" {
				continue
			}
			w.Header().Add(k, v)
		}
	}
	// ensure the response is treated as a streaming SSE response
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(resp.StatusCode)

	flusher, ok := w.(http.Flusher)
	if !ok {
		io.Copy(w, resp.Body)
		return
	}

	// stream line-by-line so SSE chunks are forwarded immediately
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		fmt.Printf("Line received %s\n", string(line))
		if _, err := w.Write(append(line, '\n')); err != nil {
			return
		}
		flusher.Flush()
	}
}
