---
name: apple-reminders
description: "Apple Reminders via remindctl: add, list, complete."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [Reminders, tasks, todo, macOS, Apple]
prerequisites:
  commands: [remindctl]
---

# Apple Reminders

Use `remindctl` to manage Apple Reminders directly from the terminal. Tasks sync across all Apple devices via iCloud.

## Prerequisites

- **macOS** with Reminders.app
- Install: `brew install steipete/tap/remindctl`
- Grant Reminders permission when prompted
- Check: `remindctl status` / Request: `remindctl authorize`

## When to Use

- User mentions "reminder" or "Reminders app"
- Creating personal to-dos with due dates that sync to iOS
- Managing Apple Reminders lists
- User wants tasks to appear on their iPhone/iPad

## When NOT to Use

- Scheduling agent alerts → use the cronjob tool instead
- Calendar events → use Apple Calendar or Google Calendar
- Project task management → use GitHub Issues, Notion, etc.
- If user says "remind me" but means an agent alert → clarify first

## Quick Reference

### View Reminders

```bash
remindctl                    # Today's reminders
remindctl today              # Today
remindctl tomorrow           # Tomorrow
remindctl week               # This week
remindctl overdue            # Past due
remindctl all                # Everything
remindctl 2026-01-04         # Specific date
```

### Manage Lists

```bash
remindctl list               # List all lists
remindctl list Work          # Show specific list
remindctl list Projects --create    # Create list
remindctl list Work --delete        # Delete list
```

### Create Reminders

```bash
remindctl add "Buy milk"
remindctl add --title "Call mom" --list Personal --due tomorrow
remindctl add --title "Meeting prep" --due "2026-02-15 09:00"
```

### Edit Reminders

```bash
remindctl edit 4A83 --title "New title"            # Rename
remindctl edit 4A83 --due "2026-05-17 22:00"       # Set due date
remindctl edit 4A83 --alarm "2026-05-17 22:00"     # Set notification time (typically = due)
remindctl edit 4A83 --notes "Replacement notes"    # Overwrite notes
remindctl edit 4A83 --list "Today"                 # Move to a different list
remindctl edit 4A83 --clear-due                    # Clear due date
remindctl edit 4A83 --complete                     # Same as `remindctl complete 4A83`
```

`<id>` is an ID prefix from `remindctl all --json` (first 4+ chars of the UUID is usually unique) or an index from `remindctl show`.

### Complete / Delete

```bash
remindctl complete 1 2 3          # Complete by ID
remindctl delete 4A83 --force     # Delete by ID
```

### Output Formats

```bash
remindctl today --json       # JSON for scripting
remindctl today --plain      # TSV format
remindctl today --quiet      # Counts only
```

## Date Formats

Accepted by `--due` and date filters:
- `today`, `tomorrow`, `yesterday`
- `YYYY-MM-DD`
- `YYYY-MM-DD HH:mm`
- ISO 8601 (`2026-01-04T12:34:56Z`)

## Rules

1. When user says "remind me", clarify: Apple Reminders (syncs to phone) vs agent cronjob alert
2. Always confirm reminder content and due date before creating
3. Use `--json` for programmatic parsing

## AppleScript Fallback

For the rare case `remindctl edit` can't cover (e.g. complex custom recurrence patterns), drop to osascript. **Gotcha:** the `id` returned by `remindctl --json` is a bare UUID, but AppleScript's `id` property is prefixed with `x-apple-reminder://`. Without the prefix, `first reminder whose id is "..."` silently fails — `delete` returns OK as a no-op, `set` errors with "Invalid index".

```bash
osascript <<'EOF'
tell application "Reminders"
  set r to first reminder whose id is "x-apple-reminder://EC0B8E91-782B-47A6-8D6A-778CDDA0D953"
  set name of r to "New title"
end tell
EOF
```
