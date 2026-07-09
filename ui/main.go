package main

import (
	"bufio"
	"database/sql"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

//go:embed index.html
var staticFS embed.FS

var (
	listenAddr = flag.String("listen", ":8081", "address to listen on")
	backendURL = flag.String("backend", "http://localhost:8080", "llama-server base URL")
	dbPath     = flag.String("db", "routed-experts.db", "SQLite database path")
)

var db *sql.DB

type routed_expert_event struct {
	Layer     int      `json:"layer"`
	TokenID   int      `json:"token_id"`
	Token     string   `json:"token"`
	SessionID string   `json:"session_id"`
	Experts   []int    `json:"experts"`
	Backend   struct {
		FFNDown    string `json:"ffn_down_exps"`
		FFNGateUp  string `json:"ffn_gate_up_exps"`
		FFNUp      string `json:"ffn_up_exps"`
		FFNGate    string `json:"ffn_gate_exps"`
	} `json:"backend"`
}

type save_session_request struct {
	Name     string                `json:"name"`
	SessionID string               `json:"session_id"`
	Events   []routed_expert_event `json:"events"`
}

type session_summary struct {
	ID        int64  `json:"id"`
	Name      string `json:"name"`
	SessionID string `json:"session_id"`
	CreatedAt string `json:"created_at"`
	EventCount int64 `json:"event_count"`
}

func main() {
	flag.Parse()

	indexHTML, err := staticFS.ReadFile("index.html")
	if err != nil {
		log.Fatalf("failed to read index.html: %v", err)
	}

	if err := initDB(); err != nil {
		log.Fatalf("failed to open database: %v", err)
	}
	defer db.Close()

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
	mux.HandleFunc("/v1/moe/save-session", handleSaveSession)
	mux.HandleFunc("/v1/moe/sessions", handleListSessions)
	mux.HandleFunc("/v1/moe/sessions/", handleGetSession)

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
	log.Printf("database: %s", *dbPath)
	log.Fatal(server.ListenAndServe())
}

func initDB() error {
	var err error
	db, err = sql.Open("sqlite", *dbPath)
	if err != nil {
		return err
	}

	schema := `
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS routed_experts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_db_id INTEGER NOT NULL,
    layer_id INTEGER NOT NULL,
    token_id INTEGER NOT NULL,
    token_text TEXT,
    session_id TEXT NOT NULL,
    experts TEXT NOT NULL,
    backend_ffn_down_exps TEXT NOT NULL,
    backend_ffn_gate_up_exps TEXT NOT NULL,
    backend_ffn_up_exps TEXT NOT NULL,
    backend_ffn_gate_exps TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_db_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_routed_experts_session_db_id ON routed_experts(session_db_id);
CREATE INDEX IF NOT EXISTS idx_routed_experts_session_id ON routed_experts(session_id);
`
	_, err = db.Exec(schema)
	if err != nil {
		return err
	}

	// migrate existing databases created before token_text was added
	_, _ = db.Exec(`ALTER TABLE routed_experts ADD COLUMN token_text TEXT IF NOT EXISTS`)

	return nil
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte("index"))
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
			if k == "Content-Length" {
				continue
			}
			w.Header().Add(k, v)
		}
	}
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

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		fmt.Printf("line received %s\n", string(line))
		if _, err := w.Write(append(line, '\n')); err != nil {
			return
		}
		flusher.Flush()
	}
}

func handleSaveSession(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req save_session_request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Name == "" {
		req.Name = req.SessionID
	}

	tx, err := db.Begin()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer tx.Rollback()

	res, err := tx.Exec(
		"INSERT INTO sessions (name, session_id) VALUES (?, ?)",
		req.Name, req.SessionID,
	)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	sessionDBID, err := res.LastInsertId()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	stmt, err := tx.Prepare(`
		INSERT INTO routed_experts (
			session_db_id, layer_id, token_id, token_text, session_id, experts,
			backend_ffn_down_exps, backend_ffn_gate_up_exps, backend_ffn_up_exps, backend_ffn_gate_exps
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer stmt.Close()

	for _, ev := range req.Events {
		expertsJSON, err := json.Marshal(ev.Experts)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		_, err = stmt.Exec(
			sessionDBID,
			ev.Layer,
			ev.TokenID,
			ev.Token,
			ev.SessionID,
			string(expertsJSON),
			ev.Backend.FFNDown,
			ev.Backend.FFNGateUp,
			ev.Backend.FFNUp,
			ev.Backend.FFNGate,
		)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	if err := tx.Commit(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": sessionDBID,
		"name": req.Name,
		"session_id": req.SessionID,
	})
}

func handleListSessions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	rows, err := db.Query(`
		SELECT s.id, s.name, s.session_id, s.created_at, COUNT(e.id) as event_count
		FROM sessions s
		LEFT JOIN routed_experts e ON e.session_db_id = s.id
		GROUP BY s.id
		ORDER BY s.created_at DESC
	`)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var sessions []session_summary
	for rows.Next() {
		var s session_summary
		if err := rows.Scan(&s.ID, &s.Name, &s.SessionID, &s.CreatedAt, &s.EventCount); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		sessions = append(sessions, s)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(sessions)
}

func handleGetSession(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	prefix := "/v1/moe/sessions/"
	if !strings.HasPrefix(r.URL.Path, prefix) {
		http.NotFound(w, r)
		return
	}
	idStr := strings.TrimPrefix(r.URL.Path, prefix)
	if idStr == "" {
		http.NotFound(w, r)
		return
	}

	var sessionID string
	var name string
	var createdAt string
	err := db.QueryRow("SELECT session_id, name, created_at FROM sessions WHERE id = ?", idStr).Scan(&sessionID, &name, &createdAt)
	if err == sql.ErrNoRows {
		http.NotFound(w, r)
		return
	} else if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	rows, err := db.Query(`
		SELECT layer_id, token_id, token_text, session_id, experts,
			backend_ffn_down_exps, backend_ffn_gate_up_exps, backend_ffn_up_exps, backend_ffn_gate_exps
		FROM routed_experts
		WHERE session_db_id = ?
		ORDER BY id ASC
	`, idStr)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var events []routed_expert_event
	for rows.Next() {
		var ev routed_expert_event
		var expertsJSON string
		if err := rows.Scan(
			&ev.Layer, &ev.TokenID, &ev.Token, &ev.SessionID, &expertsJSON,
			&ev.Backend.FFNDown, &ev.Backend.FFNGateUp, &ev.Backend.FFNUp, &ev.Backend.FFNGate,
		); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := json.Unmarshal([]byte(expertsJSON), &ev.Experts); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		events = append(events, ev)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id":         idStr,
		"name":       name,
		"session_id": sessionID,
		"created_at": createdAt,
		"events":     events,
	})
}
