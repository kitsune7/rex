package tools

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// ReminderStatus represents the state of a reminder.
type ReminderStatus string

const (
	ReminderPending   ReminderStatus = "pending"
	ReminderDelivered ReminderStatus = "delivered"
	ReminderCleared   ReminderStatus = "cleared"
)

// Reminder represents a single reminder record.
type Reminder struct {
	ID          int64
	Message     string
	DueDatetime time.Time
	CreatedAt   time.Time
	Status      ReminderStatus
}

// ReminderManager manages reminders with SQLite persistence.
type ReminderManager struct {
	mu       sync.Mutex
	db       *sql.DB
	eventBus EventEmitter
}

// NewReminderManager opens (or creates) a SQLite database at dbPath and initializes the schema.
func NewReminderManager(dbPath string, eventBus EventEmitter) (*ReminderManager, error) {
	// Ensure parent directory exists
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create db directory: %w", err)
	}

	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS reminders (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			message TEXT NOT NULL,
			due_datetime TEXT NOT NULL,
			created_at TEXT NOT NULL,
			status TEXT NOT NULL DEFAULT 'pending'
		)
	`)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("create table: %w", err)
	}

	return &ReminderManager{db: db, eventBus: eventBus}, nil
}

func (rm *ReminderManager) emitScheduleChanged() {
	if rm.eventBus != nil {
		rm.eventBus.Emit(Event{Type: "ReminderScheduleChanged"})
	}
}

// Close closes the underlying database connection.
func (rm *ReminderManager) Close() error {
	return rm.db.Close()
}

// CreateReminder creates a new pending reminder and returns its ID.
func (rm *ReminderManager) CreateReminder(message string, dueAt time.Time) (int64, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	now := time.Now()
	result, err := rm.db.Exec(
		"INSERT INTO reminders (message, due_datetime, created_at, status) VALUES (?, ?, ?, ?)",
		message, dueAt.Format(time.RFC3339), now.Format(time.RFC3339), string(ReminderPending),
	)
	if err != nil {
		return 0, fmt.Errorf("insert reminder: %w", err)
	}

	id, err := result.LastInsertId()
	if err != nil {
		return 0, err
	}

	rm.emitScheduleChanged()
	return id, nil
}

// GetReminder retrieves a single reminder by ID.
func (rm *ReminderManager) GetReminder(id int64) (*Reminder, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	return rm.scanReminder(rm.db.QueryRow("SELECT id, message, due_datetime, created_at, status FROM reminders WHERE id = ?", id))
}

// ListReminders returns all pending reminders ordered by due time.
func (rm *ReminderManager) ListReminders() ([]Reminder, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	return rm.queryReminders("SELECT id, message, due_datetime, created_at, status FROM reminders WHERE status = ? ORDER BY due_datetime", string(ReminderPending))
}

// ListRemindersByStatus returns reminders filtered by status.
func (rm *ReminderManager) ListRemindersByStatus(status ReminderStatus) ([]Reminder, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	return rm.queryReminders("SELECT id, message, due_datetime, created_at, status FROM reminders WHERE status = ? ORDER BY due_datetime", string(status))
}

// GetDueReminders returns pending reminders whose due_datetime <= now.
func (rm *ReminderManager) GetDueReminders() ([]Reminder, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	now := time.Now().Format(time.RFC3339)
	return rm.queryReminders(
		"SELECT id, message, due_datetime, created_at, status FROM reminders WHERE status = ? AND due_datetime <= ? ORDER BY due_datetime",
		string(ReminderPending), now,
	)
}

// GetNextPendingTime returns the earliest due_datetime among pending reminders.
func (rm *ReminderManager) GetNextPendingTime() (*time.Time, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	var dtStr sql.NullString
	err := rm.db.QueryRow(
		"SELECT MIN(due_datetime) FROM reminders WHERE status = ?",
		string(ReminderPending),
	).Scan(&dtStr)
	if err != nil {
		return nil, err
	}
	if !dtStr.Valid || dtStr.String == "" {
		return nil, nil
	}
	t, err := time.Parse(time.RFC3339, dtStr.String)
	if err != nil {
		return nil, err
	}
	return &t, nil
}

// UpdateReminder updates a reminder's message and/or due time.
func (rm *ReminderManager) UpdateReminder(id int64, message *string, dueAt *time.Time) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	var sets []string
	var args []any

	if message != nil {
		sets = append(sets, "message = ?")
		args = append(args, *message)
	}
	if dueAt != nil {
		sets = append(sets, "due_datetime = ?")
		args = append(args, dueAt.Format(time.RFC3339))
	}
	if len(sets) == 0 {
		return nil
	}

	args = append(args, id)
	query := fmt.Sprintf("UPDATE reminders SET %s WHERE id = ?", joinStrings(sets, ", "))
	result, err := rm.db.Exec(query, args...)
	if err != nil {
		return fmt.Errorf("update reminder: %w", err)
	}

	rows, _ := result.RowsAffected()
	if rows == 0 {
		return fmt.Errorf("reminder %d not found", id)
	}

	if dueAt != nil {
		rm.emitScheduleChanged()
	}
	return nil
}

// DeleteReminder removes a reminder by ID. Returns true if it was deleted.
func (rm *ReminderManager) DeleteReminder(id int64) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	result, err := rm.db.Exec("DELETE FROM reminders WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("delete reminder: %w", err)
	}
	rows, _ := result.RowsAffected()
	if rows == 0 {
		return fmt.Errorf("reminder %d not found", id)
	}

	rm.emitScheduleChanged()
	return nil
}

// ClearReminder marks a reminder as cleared.
func (rm *ReminderManager) ClearReminder(id int64) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	result, err := rm.db.Exec("UPDATE reminders SET status = ? WHERE id = ?", string(ReminderCleared), id)
	if err != nil {
		return fmt.Errorf("clear reminder: %w", err)
	}
	rows, _ := result.RowsAffected()
	if rows == 0 {
		return fmt.Errorf("reminder %d not found", id)
	}
	return nil
}

// SnoozeReminder updates the due time and resets status to pending.
func (rm *ReminderManager) SnoozeReminder(id int64, duration time.Duration) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	// Get current due time
	var dtStr string
	err := rm.db.QueryRow("SELECT due_datetime FROM reminders WHERE id = ?", id).Scan(&dtStr)
	if err != nil {
		return fmt.Errorf("reminder %d not found", id)
	}

	t, err := time.Parse(time.RFC3339, dtStr)
	if err != nil {
		return err
	}

	newDue := t.Add(duration)
	_, err = rm.db.Exec(
		"UPDATE reminders SET due_datetime = ?, status = ? WHERE id = ?",
		newDue.Format(time.RFC3339), string(ReminderPending), id,
	)
	if err != nil {
		return fmt.Errorf("snooze reminder: %w", err)
	}

	rm.emitScheduleChanged()
	return nil
}

// scanReminder scans a single row into a Reminder. Returns nil if no rows.
func (rm *ReminderManager) scanReminder(row *sql.Row) (*Reminder, error) {
	var r Reminder
	var dueStr, createdStr, statusStr string
	err := row.Scan(&r.ID, &r.Message, &dueStr, &createdStr, &statusStr)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	r.DueDatetime, _ = time.Parse(time.RFC3339, dueStr)
	r.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
	r.Status = ReminderStatus(statusStr)
	return &r, nil
}

// queryReminders runs a query and returns a slice of reminders.
func (rm *ReminderManager) queryReminders(query string, args ...any) ([]Reminder, error) {
	rows, err := rm.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []Reminder
	for rows.Next() {
		var r Reminder
		var dueStr, createdStr, statusStr string
		if err := rows.Scan(&r.ID, &r.Message, &dueStr, &createdStr, &statusStr); err != nil {
			return nil, err
		}
		r.DueDatetime, _ = time.Parse(time.RFC3339, dueStr)
		r.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
		r.Status = ReminderStatus(statusStr)
		result = append(result, r)
	}
	return result, rows.Err()
}

func joinStrings(ss []string, sep string) string {
	if len(ss) == 0 {
		return ""
	}
	result := ss[0]
	for _, s := range ss[1:] {
		result += sep + s
	}
	return result
}
