#%%
from awpy import Demo


demo_file = "C:\\Users\\Rickard\\Downloads\\blast-open-london-2025-finals-vitality-vs-mouz-bo3-fktnIE0VMDsgMDWkVX_HmW\\vitality-vs-mouz-m1-mirage.dem"

dem = Demo(demo_file)

dem.parse(player_props=['armor_value', 'has_helmet', 'has_defuser', 'inventory'])

#%%
print(dem)

# %%
pickup=dem.events['item_pickup'].to_pandas()
#%%
print(pickup)

# %%
# Let's investigate what data is available in the parsed demo object,
# especially regarding player inventory at specific ticks.

ticks_df = dem.ticks.to_pandas()

print("--- Tick Data Investigation ---")
print(f"Shape of ticks_df: {ticks_df.shape}")
print("Columns available in the ticks dataframe:")
print(ticks_df.columns)

if not ticks_df.empty:
    print("\n--- Sample Tick Row ---")
    # Print a sample row to see the structure
    print(ticks_df.iloc[5000])

    if 'inventory' in ticks_df.columns:
        print("\n--- Inventory Column Found! ---")
        # Find a row where inventory is not None or empty
        inventory_sample = ticks_df[ticks_df['inventory'].notna()]
        if not inventory_sample.empty:
            print("Sample inventory object from the first non-empty inventory row:")
            print(inventory_sample.iloc[0]['inventory'])
        else:
            print("The 'inventory' column exists, but all values are empty/NaN.")
    else:
        print("\n--- 'inventory' Column NOT FOUND in ticks data. ---")
        print("This is likely why the equipment extraction is failing.")
        print("The version of `awpy` or the demo file might not support this feature.")

print("\n--- Available Events ---")
print("List of all event types available in the demo:")
print(list(dem.events.keys()))

# %%
# Simple inventory samples - LIMITED DATA
print("\n--- INVENTORY SAMPLES (QUICK) ---")

if 'inventory' in ticks_df.columns:
    # Take only first 1000 rows to speed things up
    sample_df = ticks_df.head(1000)
    print(f"Working with first 1000 ticks instead of {len(ticks_df)} total ticks")
    
    # Get some non-empty inventories from the sample
    non_empty_inventories = sample_df[sample_df['inventory'].notna() & (sample_df['inventory'].astype(str) != '[]')]
    
    if not non_empty_inventories.empty:
        print(f"Found {len(non_empty_inventories)} ticks with inventories in first 1000 ticks")
        
        # Show first 3 unique inventory examples
        unique_inventories = non_empty_inventories['inventory'].drop_duplicates().head(30)
        
        for i, inventory in enumerate(unique_inventories):
            print(f"\nExample {i+1}: {inventory}")
            print(f"Type: {type(inventory)}")
            
            if isinstance(inventory, list):
                print(f"Items: {len(inventory)} items")
                for item in inventory:
                    print(f"  - {item}")
    else:
        print("No inventories found in first 1000 ticks. Try later in the game...")
        # Try a different sample from middle of the game
        middle_sample = ticks_df.iloc[10000:11000] if len(ticks_df) > 11000 else ticks_df.tail(1000)
        non_empty_inventories = middle_sample[middle_sample['inventory'].notna() & (middle_sample['inventory'].astype(str) != '[]')]
        
        if not non_empty_inventories.empty:
            print(f"Found {len(non_empty_inventories)} inventories in middle sample")
            unique_inventories = non_empty_inventories['inventory'].drop_duplicates().head(30)
            
            for i, inventory in enumerate(unique_inventories):
                print(f"\nExample {i+1}: {inventory}")
                if isinstance(inventory, list):
                    for item in inventory:
                        print(f"  - {item}")
else:
    print("No 'inventory' column found.")

# %%
