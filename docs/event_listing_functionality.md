# Event Listing Functionality

This document describes the new event listing functionality added to the CS2 demo processing pipeline.

## Overview

The pipeline now supports listing and processing events directly from the HLTV events archive page. This allows you to:

1. **List events** from HLTV archive with customizable filters
2. **Choose how many events** to process
3. **Process multiple events** automatically from the archive

## New Features

### Event Listing

List events from HLTV archive:

```bash
# List 20 events (default)
python -m src.workflows.demo_pipeline --list-events

# List 10 events with custom filters
python -m src.workflows.demo_pipeline --list-events --max-events 10 --prize-min 100000

# List events with specific criteria
python -m src.workflows.demo_pipeline --list-events \
    --max-events 15 \
    --event-type INTLLAN \
    --prize-min 50000 \
    --prize-max 1000000
```

### Processing Events from Archive

Process demos from events listed in the archive:

```bash
# Process first 3 events from archive (max 1 event actually processed)
python -m src.workflows.demo_pipeline --from-archive \
    --max-events 3 \
    --max-events-to-process 1 \
    --max-demos 5

# Process all events from archive with filters
python -m src.workflows.demo_pipeline --from-archive \
    --max-events 10 \
    --prize-min 200000 \
    --max-demos 10
```

## Command Line Options

### Archive-Specific Options

- `--list-events`: List events from HLTV archive without processing
- `--from-archive`: Process events from HLTV archive
- `--max-events NUMBER`: Maximum number of events to list from archive (default: 20)
- `--max-events-to-process NUMBER`: Maximum number of events to actually process from archive
- `--event-type TYPE`: Event type filter (default: INTLLAN)
- `--prize-min AMOUNT`: Minimum prize pool in USD (default: 28807)
- `--prize-max AMOUNT`: Maximum prize pool in USD (default: 2000000)

### Existing Options

- `--max-demos NUMBER`: Maximum number of demos per event
- `--temp-dir PATH`: Temporary directory for downloads
- `--keep-demos`: Keep original .dem files after processing

## Filter Parameters

### Event Types

- `INTLLAN`: International LAN events (default)
- `LAN`: LAN events
- `Online`: Online events

### Prize Pool Filters

- `--prize-min`: Minimum prize pool in USD (default: $28,807)
- `--prize-max`: Maximum prize pool in USD (default: $2,000,000)

## Examples

### Example 1: List Recent High-Value Events

```bash
python -m src.workflows.demo_pipeline --list-events \
    --max-events 5 \
    --prize-min 500000
```

This will list the first 5 events with prize pools of at least $500,000.

### Example 2: Process Specific Number of Events

```bash
python -m src.workflows.demo_pipeline --from-archive \
    --max-events 10 \
    --max-events-to-process 2 \
    --max-demos 3
```

This will:

1. List the first 10 events from the archive
2. Process only the first 2 events from that list
3. Process maximum 3 demos per event

### Example 3: Process All Listed Events

```bash
python -m src.workflows.demo_pipeline --from-archive \
    --max-events 5 \
    --max-demos 10
```

This will process all 5 listed events with up to 10 demos each.

## Implementation Details

### EventRef Class

```python
@dataclass(frozen=True)
class EventRef:
    event_id: int
    name: str
    date_range: str
    prize_pool: str
    location: str
    url: str  # absolute URL to the event page
```

### Key Functions

- `list_events_from_archive()`: Scrapes events from HLTV archive
- `process_events_from_archive()`: Lists and processes events from archive
- Enhanced command line interface with new options

## Error Handling

The system includes robust error handling:

- Skips problematic events during parsing
- Continues processing if individual events fail
- Provides detailed progress information
- Respects HLTV rate limits

## Rate Limiting

The implementation is polite to HLTV:

- 1-second delays between archive page requests
- 0.5-second delays between match page requests
- Uses proper User-Agent identification
- Respects HTTP error codes and retries appropriately
