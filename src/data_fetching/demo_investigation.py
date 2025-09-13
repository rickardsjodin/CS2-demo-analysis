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


