import json
import matplotlib.pyplot as plt
import numpy as np

# Load the two datasets
with open('player_velocity_data_v1.json', 'r') as f:
    data_0_offset = json.load(f)

with open('player_velocity_data.json', 'r') as f:
    data_10s_offset = json.load(f)

# Extract player info
players_0 = data_0_offset['player_info']
players_10s = data_10s_offset['player_info']

# Find common players
common_player_ids = set(players_0.keys()) & set(players_10s.keys())

# Prepare data for plotting
avg_vel_0 = []
avg_time_0 = []
avg_vel_10s = []
avg_time_10s = []
player_names = []

for player_id in common_player_ids:
    p0 = players_0[player_id]
    p10 = players_10s[player_id]

    if p0['round_count'] < 10 or p10['round_count'] < 10 or p0['total_vel_xy'] < 10:
        continue
    
    # Calculate averages
    avg_vel_0.append((p0['total_vel_xy'] / p0['round_count']))
    avg_time_0.append((p0['total_time_alive'] / p0['round_count']))
    
    avg_vel_10s.append(p10['total_vel_xy'] / p10['round_count'])
    avg_time_10s.append(p10['total_time_alive'] / p10['round_count'] + 20)
    
    player_names.append(p0['name'])

# Calculate R² for each dataset
from scipy import stats

# R² for 0s offset
slope_0, intercept_0, r_value_0, p_value_0, std_err_0 = stats.linregress(avg_vel_0, avg_time_0)
r_squared_0 = r_value_0 ** 2

# R² for 10s offset
slope_10s, intercept_10s, r_value_10s, p_value_10s, std_err_10s = stats.linregress(avg_vel_10s, avg_time_10s)
r_squared_10s = r_value_10s ** 2

# Create the plot
plt.figure(figsize=(12, 8))

# Plot points for 0 offset
plt.scatter(avg_vel_0, avg_time_0, c='blue', s=100, alpha=0.6, label=f'0s offset (R²={r_squared_0:.4f})', zorder=3)

# Plot points for 10s offset
plt.scatter(avg_vel_10s, avg_time_10s, c='red', s=100, alpha=0.6, label=f'10s offset (R²={r_squared_10s:.4f})', zorder=3)

# Draw lines connecting the same player's points
# for i in range(len(player_names)):
#     plt.plot([avg_vel_0[i], avg_vel_10s[i]], 
#              [avg_time_0[i], avg_time_10s[i]], 
#              'gray', alpha=0.3, linewidth=1, zorder=1)

# Add player name labels
# for i in range(len(player_names)):
#     plt.text(avg_vel_0[i], avg_time_0[i], player_names[i], fontsize=8, alpha=0.7, ha='left', )

# Add labels and title
plt.xlabel('Average XY Velocity (units/second)', fontsize=12)
plt.ylabel('Average Time Alive per Round (seconds)', fontsize=12)
plt.title('Player Velocity vs Time Alive: 0s Offset vs 10s Offset', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('velocity_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'velocity_comparison.png'")

# Print some statistics
print(f"\nAnalyzed {len(common_player_ids)} common players")
print(f"\n0s Offset - Avg velocity: {np.mean(avg_vel_0):.2f}, Avg time alive: {np.mean(avg_time_0):.2f}, R²: {r_squared_0:.4f}")
print(f"10s Offset - Avg velocity: {np.mean(avg_vel_10s):.2f}, Avg time alive: {np.mean(avg_time_10s):.2f}, R²: {r_squared_10s:.4f}")


# Create second plot with aligned means
avg_vel_0_aligned = np.array(avg_vel_0) - np.mean(avg_vel_0)
avg_vel_10s_aligned = np.array(avg_vel_10s) - np.mean(avg_vel_10s)

# Calculate R² for aligned datasets
slope_0_aligned, intercept_0_aligned, r_value_0_aligned, p_value_0_aligned, std_err_0_aligned = stats.linregress(avg_vel_0_aligned, avg_time_0)
r_squared_0_aligned = r_value_0_aligned ** 2

slope_10s_aligned, intercept_10s_aligned, r_value_10s_aligned, p_value_10s_aligned, std_err_10s_aligned = stats.linregress(avg_vel_10s_aligned, avg_time_10s)
r_squared_10s_aligned = r_value_10s_aligned ** 2

plt.figure(figsize=(12, 8))

# Plot points for 0 offset
plt.scatter(avg_vel_0_aligned, avg_time_0, c='blue', s=100, alpha=0.6, label=f'0s offset (R²={r_squared_0_aligned:.4f})', zorder=3)

# Plot points for 10s offset
plt.scatter(avg_vel_10s_aligned, avg_time_10s, c='red', s=100, alpha=0.6, label=f'10s offset (R²={r_squared_10s_aligned:.4f})', zorder=3)

# Add labels and title
plt.xlabel('Average XY Velocity - Mean Centered (units/second)', fontsize=12)
plt.ylabel('Average Time Alive per Round (seconds)', fontsize=12)
plt.title('Player Velocity vs Time Alive (Mean-Aligned): 0s Offset vs 10s Offset', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

# Adjust layout and save
plt.tight_layout()
plt.savefig('velocity_comparison_aligned.png', dpi=300, bbox_inches='tight')
print("\nAligned plot saved as 'velocity_comparison_aligned.png'")

plt.show()
