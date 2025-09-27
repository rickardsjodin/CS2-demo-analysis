#%%
import pandas as pd
from src.ml.train_win_probability_model import load_and_prepare_data


_, _, _, df = load_and_prepare_data(data_file=None)

#%%

base_input = {
    "cts_alive":      {"value": 5,   "bin_size": 0},
    "ts_alive":       {"value": 5,   "bin_size": 0},
    "ct_armor":       {"value": 4,   "bin_size": 1},
    "t_armor":        {"value": 4,   "bin_size": 1},
    "ct_helmets":     {"value": 4,   "bin_size": 1},
    "t_helmets":      {"value": 4,   "bin_size": 1},
    "ct_main_weapons":{"value": 4,   "bin_size": 1},
    "t_main_weapons": {"value": 4,   "bin_size": 1},
    "bomb_planted":   {"value": 0,   "bin_size": 0},
    "bomb_time_left": {"value": 0,   "bin_size": 0},
    "hp_ct":          {"value": 500, "bin_size": 50},
    "hp_t":           {"value": 500, "bin_size": 50},
    "round_time_left":{"value": 100,  "bin_size": 15},
    "defusers":       {"value": 3,   "bin_size": -1},
    "ct_flashes":     {"value": 3,   "bin_size": -1},
    "ct_he_nades":    {"value": 2,   "bin_size": -1},
    "ct_molotovs":    {"value": 1,   "bin_size": -1},
    "ct_smokes":      {"value": 2,   "bin_size": -1},
    "t_flashes":      {"value": 3,   "bin_size": -1},
    "t_he_nades":     {"value": 2,   "bin_size": -1},
    "t_molotovs":     {"value": 1,   "bin_size": -1},
    "t_smokes":       {"value": 2,   "bin_size": -1},
}

# Slice df by binning the base input
mask = pd.Series(True, index=df.index)
for key, cfg in base_input.items():
    v, bs = cfg['value'], cfg['bin_size']
    if bs < 0:
        continue
    mask &= df[key].between(v - bs, v + bs)

df_filtered = df[mask]

print(len(df_filtered))

#%%

p_ct = len(df_filtered[df_filtered['winner'] == 'ct']) / len(df_filtered)
print(f"P CT: {p_ct}")



# %%
print(df_filtered)

# %%
