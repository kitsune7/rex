// Package reminders provides a SQLite-backed implementation of
// core.ReminderStore. Schema parity with src/agent/tools/reminder.py:
//
//	CREATE TABLE reminders (
//	    id INTEGER PRIMARY KEY AUTOINCREMENT,
//	    message TEXT NOT NULL,
//	    due_datetime TEXT NOT NULL,      -- RFC3339 / ISO-8601
//	    created_at TEXT NOT NULL,
//	    status TEXT NOT NULL DEFAULT 'pending'
//	)
//
// due_datetime and created_at are stored as ISO-8601 strings rather than
// sqlite DATETIME so the format matches the Python implementation byte-for-byte.
package reminders

import (
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "github.com/mattn/go-sqlite3" // register sqlite3 driver

	"rex/internal/core"
)

const (
	statusPending   = "pending"
	statusDelivered = "delivered"
	statusCleared   = "cleared"

	// timeLayout mirrors Python's datetime.isoformat() with microsecond precision,
	// e.g. "2026-04-30T09:05:00". Go's time.RFC3339Nano produces an equivalent
	// round-trippable form ("2026-04-30T09:05:00Z"); we strip the timezone on
	// write so comparisons line up with naive Python datetimes, and parse on
	// read with a permissive set of layouts.
	timeLayout = "2006-01-02T15:04:05.999999"
)

// Store is a SQLite-backed ReminderStore. Safe for concurrent use.
type Store struct {
	db  *sql.DB
	bus *core.EventBus
}

// Open opens (or creates) the SQLite database at path and applies the schema.
// Pass a non-nil EventBus to have the store emit ReminderScheduleChanged on
// mutations.
func Open(path string, bus *core.EventBus) (*Store, error) {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("creating reminder db directory: %w", err)
		}
	}

	// _busy_timeout=5000 prevents "database is locked" errors under light
	// concurrent writes — sqlite will retry for up to 5 seconds.
	dsn := fmt.Sprintf("file:%s?_busy_timeout=5000&_journal_mode=WAL&_foreign_keys=on", path)
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("opening sqlite: %w", err)
	}

	// mattn/go-sqlite3 serializes writes per-connection, so a single connection
	// avoids SQLITE_BUSY errors from concurrent goroutines in tests.
	db.SetMaxOpenConns(1)

	if _, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS reminders (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			message TEXT NOT NULL,
			due_datetime TEXT NOT NULL,
			created_at TEXT NOT NULL,
			status TEXT NOT NULL DEFAULT 'pending'
		)
	`); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("initialising schema: %w", err)
	}

	return &Store{db: db, bus: bus}, nil
}

// Close releases the underlying database connection.
func (s *Store) Close() error {
	return s.db.Close()
}

func (s *Store) emitScheduleChanged() {
	if s.bus != nil {
		s.bus.Emit(core.ReminderScheduleChanged{BaseEvent: core.NewBaseEvent()})
	}
}

// CreateReminder inserts a new pending reminder and returns its ID.
func (s *Store) CreateReminder(message string, dueAt time.Time) (int64, error) {
	now := time.Now()
	res, err := s.db.Exec(
		`INSERT INTO reminders (message, due_datetime, created_at, status) VALUES (?, ?, ?, ?)`,
		message, formatTime(dueAt), formatTime(now), statusPending,
	)
	if err != nil {
		return 0, fmt.Errorf("inserting reminder: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("fetching inserted id: %w", err)
	}
	s.emitScheduleChanged()
	return id, nil
}

// ListReminders returns all reminders ordered by due_datetime.
func (s *Store) ListReminders() ([]core.Reminder, error) {
	return s.query(`SELECT id, message, due_datetime, created_at, status FROM reminders ORDER BY due_datetime`)
}

// GetDueReminders returns pending reminders whose due time has passed.
func (s *Store) GetDueReminders() ([]core.Reminder, error) {
	return s.query(
		`SELECT id, message, due_datetime, created_at, status
		   FROM reminders
		  WHERE status = ? AND due_datetime <= ?
		  ORDER BY due_datetime`,
		statusPending, formatTime(time.Now()),
	)
}

// GetNextPendingTime returns the earliest due time among pending reminders,
// or nil if there are none.
func (s *Store) GetNextPendingTime() (*time.Time, error) {
	var ns sql.NullString
	err := s.db.QueryRow(
		`SELECT MIN(due_datetime) FROM reminders WHERE status = ?`, statusPending,
	).Scan(&ns)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("querying next pending time: %w", err)
	}
	if !ns.Valid {
		return nil, nil
	}
	t, err := parseTime(ns.String)
	if err != nil {
		return nil, fmt.Errorf("parsing next pending time: %w", err)
	}
	return &t, nil
}

// UpdateReminder updates the message and/or due time of a reminder. Fields
// that are nil are left unchanged. Emits ReminderScheduleChanged only when
// the due time was touched.
func (s *Store) UpdateReminder(id int64, message *string, dueAt *time.Time) error {
	if message == nil && dueAt == nil {
		return nil
	}

	query := `UPDATE reminders SET `
	args := []any{}
	if message != nil {
		query += `message = ?`
		args = append(args, *message)
	}
	if dueAt != nil {
		if len(args) > 0 {
			query += `, `
		}
		query += `due_datetime = ?`
		args = append(args, formatTime(*dueAt))
	}
	query += ` WHERE id = ?`
	args = append(args, id)

	res, err := s.db.Exec(query, args...)
	if err != nil {
		return fmt.Errorf("updating reminder: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("checking update result: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("reminder %d not found", id)
	}

	if dueAt != nil {
		s.emitScheduleChanged()
	}
	return nil
}

// DeleteReminder removes a reminder by ID.
func (s *Store) DeleteReminder(id int64) error {
	res, err := s.db.Exec(`DELETE FROM reminders WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("deleting reminder: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("checking delete result: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("reminder %d not found", id)
	}
	s.emitScheduleChanged()
	return nil
}

// ClearReminder marks a reminder as cleared (acknowledged by the user).
func (s *Store) ClearReminder(id int64) error {
	return s.updateStatus(id, statusCleared)
}

// SnoozeReminder pushes the due time forward by the given duration and
// returns the reminder to pending. Emits ReminderScheduleChanged.
func (s *Store) SnoozeReminder(id int64, d time.Duration) error {
	// Fetch current due time, add the snooze offset, write back + reset status.
	var (
		dueStr string
	)
	err := s.db.QueryRow(`SELECT due_datetime FROM reminders WHERE id = ?`, id).Scan(&dueStr)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("reminder %d not found", id)
		}
		return fmt.Errorf("fetching due time: %w", err)
	}
	due, err := parseTime(dueStr)
	if err != nil {
		return fmt.Errorf("parsing due time: %w", err)
	}
	newDue := due.Add(d)

	_, err = s.db.Exec(
		`UPDATE reminders SET due_datetime = ?, status = ? WHERE id = ?`,
		formatTime(newDue), statusPending, id,
	)
	if err != nil {
		return fmt.Errorf("snoozing reminder: %w", err)
	}
	s.emitScheduleChanged()
	return nil
}

func (s *Store) updateStatus(id int64, status string) error {
	res, err := s.db.Exec(`UPDATE reminders SET status = ? WHERE id = ?`, status, id)
	if err != nil {
		return fmt.Errorf("updating status: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("checking status update: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("reminder %d not found", id)
	}
	return nil
}

// MarkDelivered flips a reminder's status to delivered. Exposed for the
// scheduler (Stage 8) but included here so the state machine is complete.
func (s *Store) MarkDelivered(id int64) error {
	return s.updateStatus(id, statusDelivered)
}

func (s *Store) query(stmt string, args ...any) ([]core.Reminder, error) {
	rows, err := s.db.Query(stmt, args...)
	if err != nil {
		return nil, fmt.Errorf("running query: %w", err)
	}
	defer rows.Close()

	var out []core.Reminder
	for rows.Next() {
		r, err := scanReminder(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating rows: %w", err)
	}
	return out, nil
}

func scanReminder(rows *sql.Rows) (core.Reminder, error) {
	var (
		id              int64
		message         string
		dueStr, cStr    string
		statusStr       string
	)
	if err := rows.Scan(&id, &message, &dueStr, &cStr, &statusStr); err != nil {
		return core.Reminder{}, fmt.Errorf("scanning row: %w", err)
	}
	due, err := parseTime(dueStr)
	if err != nil {
		return core.Reminder{}, fmt.Errorf("parsing due_datetime: %w", err)
	}
	created, err := parseTime(cStr)
	if err != nil {
		return core.Reminder{}, fmt.Errorf("parsing created_at: %w", err)
	}
	status, err := decodeStatus(statusStr)
	if err != nil {
		return core.Reminder{}, err
	}
	return core.Reminder{
		ID:        id,
		Message:   message,
		DueAt:     due,
		CreatedAt: created,
		Status:    status,
	}, nil
}

func decodeStatus(s string) (core.ReminderStatus, error) {
	switch s {
	case statusPending:
		return core.ReminderPending, nil
	case statusDelivered:
		return core.ReminderDelivered, nil
	case statusCleared:
		return core.ReminderCleared, nil
	}
	return 0, fmt.Errorf("unknown reminder status %q", s)
}

func formatTime(t time.Time) string {
	// Strip monotonic clock and drop timezone info (store as local naive time)
	// so strings sort lexically in chronological order within a single locale.
	return t.Format(timeLayout)
}

func parseTime(s string) (time.Time, error) {
	// Try the canonical layout first, then fall back to RFC3339 variants in
	// case earlier code wrote timezone-qualified times.
	layouts := []string{
		timeLayout,
		"2006-01-02T15:04:05",
		time.RFC3339Nano,
		time.RFC3339,
	}
	for _, layout := range layouts {
		if t, err := time.ParseInLocation(layout, s, time.Local); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("unrecognised time format %q", s)
}
