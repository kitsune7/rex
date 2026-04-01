// Package scheduler implements a background reminder scheduler that wakes at
// precise times when reminders are due, rather than polling.
package scheduler

import "time"

// ReminderDelivery holds information about a reminder currently being delivered.
type ReminderDelivery struct {
	ID      int64
	Message string
	DueAt   time.Time
}
