from awpy import Demo
import pandas as pd
import polars as pl

# Parse the demo file
dem = Demo('the-mongolz-vs-vitality-m1-mirage.dem')
dem.parse()

# Quick check of bomb events
bomb_events = dem.bomb_events
if bomb_events is not None and not bomb_events.empty:
    print('Bomb events found:', len(bomb_events))
    print('Columns:', bomb_events.columns.tolist())
    if len(bomb_events) > 0:
        print('\nFirst few bomb events:')
        print(bomb_events.head())
        
        # Check unique values in key columns
        if 'event_type' in bomb_events.columns:
            print('\nUnique event types:')
            print(bomb_events['event_type'].unique())
else:
    print('No bomb events found')
