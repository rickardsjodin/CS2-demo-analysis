"""
Example script demonstrating how to calculate probabilities at every tick of a round.
Includes visualization with matplotlib plots.
"""

import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from config import DEFAULT_DEMO_FILE, MODELS_DIR, VISUALIZATIONS_DIR
from src.utils.cache_utils import load_demo
from src.core.analysis import calculate_round_probabilities_all_ticks


def plot_round_probabilities(tick_probs, round_num, save_path=None):
    """
    Create a visualization of probability changes throughout a round.
    
    Args:
        tick_probs: List of tick probability dictionaries
        round_num: Round number being analyzed
        save_path: Optional path to save the plot
    """
    if not tick_probs:
        print("No data to plot")
        return
    
    # Extract data for plotting
    time_elapsed = [p['time_elapsed'] for p in tick_probs]
    ct_prob = [p['ct_win_probability'] * 100 for p in tick_probs]
    
    # Detect significant probability changes (2% or higher)
    significant_changes = []
    for i in range(1, len(tick_probs)):
        prev_tick = tick_probs[i-1]
        curr_tick = tick_probs[i]
        
        delta = abs(curr_tick['ct_win_probability'] - prev_tick['ct_win_probability']) * 100
        
        if delta >= 1.0:
            # Identify what changed
            changes = []
            
            if prev_tick['cts_alive'] != curr_tick['cts_alive']:
                diff = curr_tick['cts_alive'] - prev_tick['cts_alive']
                changes.append(f"CT: {prev_tick['cts_alive']}→{curr_tick['cts_alive']}")
            
            if prev_tick['ts_alive'] != curr_tick['ts_alive']:
                diff = curr_tick['ts_alive'] - prev_tick['ts_alive']
                changes.append(f"T: {prev_tick['ts_alive']}→{curr_tick['ts_alive']}")
            
            if prev_tick['bomb_planted'] != curr_tick['bomb_planted']:
                changes.append("Bomb Planted")
            
            if prev_tick['hp_ct'] != curr_tick['hp_ct']:
                hp_diff = curr_tick['hp_ct'] - prev_tick['hp_ct']
                if abs(hp_diff) >= 10:  # Only show significant HP changes
                    changes.append(f"CT HP: {hp_diff:+d}")
            
            if prev_tick['hp_t'] != curr_tick['hp_t']:
                hp_diff = curr_tick['hp_t'] - prev_tick['hp_t']
                if abs(hp_diff) >= 10:  # Only show significant HP changes
                    changes.append(f"T HP: {hp_diff:+d}")
            
            if changes:
                significant_changes.append({
                    'time': curr_tick['time_elapsed'],
                    'tick': curr_tick['tick'],
                    'delta': delta,
                    'prob_before': prev_tick['ct_win_probability'] * 100,
                    'prob_after': curr_tick['ct_win_probability'] * 100,
                    'changes': ', '.join(changes),
                    'state': f"{curr_tick['cts_alive']}v{curr_tick['ts_alive']}"
                })
    
    # Find key events (kills - when player counts change)
    kill_times = []
    kill_probs = []
    kill_labels = []
    prev_state = None
    
    for p in tick_probs:
        current_state = (p['cts_alive'], p['ts_alive'])
        if prev_state and current_state != prev_state:
            kill_times.append(p['time_elapsed'])
            kill_probs.append(p['ct_win_probability'] * 100)
            kill_labels.append(f"{current_state[0]}v{current_state[1]}")
        prev_state = current_state
    
    # Find bomb plant event
    plant_time = None
    plant_prob = None
    prev_planted = False
    for p in tick_probs:
        if p['bomb_planted'] and not prev_planted:
            # First tick where bomb becomes planted
            plant_time = p['time_elapsed']
            plant_prob = p['ct_win_probability'] * 100
            break
        prev_planted = p['bomb_planted']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot CT probability line
    ax.plot(time_elapsed, ct_prob, label='CT Win Probability', 
            color='#3498db', linewidth=2.5, alpha=0.9)
    
    # Fill area under curve (above/below 50%)
    ax.fill_between(time_elapsed, ct_prob, 50, where=[p >= 50 for p in ct_prob],
                    color='#3498db', alpha=0.2, interpolate=True, label='CT Favored')
    ax.fill_between(time_elapsed, ct_prob, 50, where=[p < 50 for p in ct_prob],
                    color='#e74c3c', alpha=0.2, interpolate=True, label='T Favored')
    
    # Mark kills with vertical lines and dots
    for kt, kp, label in zip(kill_times, kill_probs, kill_labels):
        ax.axvline(x=kt, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.plot(kt, kp, 'o', color='black', markersize=8, alpha=0.7, zorder=5)
        # Add state label near the kill marker
        ax.text(kt, kp + 3, label, fontsize=8, ha='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Mark significant probability changes (>=2%)
    for change in significant_changes:
        change_time = change['time']
        change_prob = change['prob_after']
        # Use star markers for significant changes
        ax.plot(change_time, change_prob, '*', color='#9b59b6', 
                markersize=14, alpha=0.8, zorder=6, 
                markeredgecolor='black', markeredgewidth=0.5,
                label='Significant Change (≥2%)' if change == significant_changes[0] else '')
    
    # Mark bomb plant with prominent vertical span and annotation
    if plant_time:
        ax.axvline(x=plant_time, color='#ff6b35', linestyle='-', 
                  alpha=0.7, linewidth=3, label='Bomb Planted', zorder=4)
        ax.plot(plant_time, plant_prob, 'D', color='#ff6b35', 
                markersize=12, alpha=0.9, zorder=6, markeredgecolor='black', markeredgewidth=1)
        # Add "BOMB PLANTED" annotation
        ax.annotate('BOMB PLANTED', 
                   xy=(plant_time, plant_prob), 
                   xytext=(plant_time, plant_prob + 15),
                   ha='center',
                   fontsize=10,
                   fontweight='bold',
                   color='#ff6b35',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='#ff6b35', linewidth=2, alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color='#ff6b35', lw=2))
    
    # 50% line
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Styling
    ax.set_xlabel('Time Elapsed (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Win Probability (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Round {round_num} - Win Probability Over Time', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(time_elapsed))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
    
    # Add final outcome text
    final_state = tick_probs[-1]
    winner = "CT" if final_state['cts_alive'] > 0 else "T"
    final_score = f"{final_state['cts_alive']}v{final_state['ts_alive']}"
    ax.text(0.98, 0.02, f"Winner: {winner}\nFinal: {final_score}", 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    
    plt.show()
    
    # Return significant changes for table display
    return significant_changes


def main():
    """Calculate and display probabilities for every tick of a specific round."""
    
    # Configuration
    DEMO_FILE = "C:\\Users\\Rickard\\Downloads\\esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj\\furia-vs-vitality-m1-overpass.dem"
    ROUND_TO_ANALYZE = 5  # Change this to analyze different rounds
    MODEL_FILE = MODELS_DIR / "ct_win_probability_xgboost_minimal.pkl"
    
    # Load the demo
    print(f"📂 Loading demo: {DEMO_FILE}")
    dem = load_demo(str(DEMO_FILE), use_cache=True)
    
    if dem is None:
        print("❌ Failed to load demo")
        return
    
    print("✅ Demo loaded successfully!")
    
    # Load the prediction model
    print(f"🤖 Loading model: {MODEL_FILE}")
    try:
        with open(MODEL_FILE, 'rb') as f:
            pred_model = pickle.load(f)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Calculate probabilities for every tick
    print(f"\n🔍 Analyzing Round {ROUND_TO_ANALYZE}...")
    tick_probs = calculate_round_probabilities_all_ticks(
        dem_file=dem,
        round_num=ROUND_TO_ANALYZE,
        pred_model=pred_model,
        tick_rate=64,
        file_name=DEMO_FILE
    )
    
    if not tick_probs:
        print(f"❌ No probabilities calculated for round {ROUND_TO_ANALYZE}")
        return
    
    # Display results
    print(f"\n✅ Calculated probabilities for {len(tick_probs)} ticks\n")
    
    # Show sample of results (first 10, every 64th tick, and last 10)
    print("=" * 100)
    print(f"{'Tick':<8} {'Time (s)':<10} {'Time Left':<12} {'CT Win %':<10} {'T Win %':<10} {'State':<15} {'Bomb':<8}")
    print("=" * 100)
    
    # First 10 ticks
    for entry in tick_probs[:10]:
        state = f"{entry['cts_alive']}v{entry['ts_alive']}"
        bomb_status = "Planted" if entry['bomb_planted'] else "No"
        print(f"{entry['tick']:<8} {entry['time_elapsed']:<10.2f} {entry['time_left']:<12.2f} "
              f"{entry['ct_win_probability']*100:<10.1f} {entry['t_win_probability']*100:<10.1f} "
              f"{state:<15} {bomb_status:<8}")
    
    if len(tick_probs) > 20:
        print("...")
        
        # Every 64th tick (1 second intervals)
        for i in range(10, len(tick_probs) - 10, 64):
            entry = tick_probs[i]
            state = f"{entry['cts_alive']}v{entry['ts_alive']}"
            bomb_status = "Planted" if entry['bomb_planted'] else "No"
            print(f"{entry['tick']:<8} {entry['time_elapsed']:<10.2f} {entry['time_left']:<12.2f} "
                  f"{entry['ct_win_probability']*100:<10.1f} {entry['t_win_probability']*100:<10.1f} "
                  f"{state:<15} {bomb_status:<8}")
        
        print("...")
        
        # Last 10 ticks
        for entry in tick_probs[-10:]:
            state = f"{entry['cts_alive']}v{entry['ts_alive']}"
            bomb_status = "Planted" if entry['bomb_planted'] else "No"
            print(f"{entry['tick']:<8} {entry['time_elapsed']:<10.2f} {entry['time_left']:<12.2f} "
                  f"{entry['ct_win_probability']*100:<10.1f} {entry['t_win_probability']*100:<10.1f} "
                  f"{state:<15} {bomb_status:<8}")
    
    print("=" * 100)
    
    # Summary statistics
    ct_probs = [p['ct_win_probability'] for p in tick_probs]
    print(f"\n📊 Summary Statistics:")
    print(f"   Average CT Win Probability: {sum(ct_probs) / len(ct_probs) * 100:.1f}%")
    print(f"   Max CT Win Probability: {max(ct_probs) * 100:.1f}%")
    print(f"   Min CT Win Probability: {min(ct_probs) * 100:.1f}%")
    print(f"   Total ticks analyzed: {len(tick_probs)}")
    print(f"   Round duration: {tick_probs[-1]['time_elapsed']:.1f} seconds")
    
    # Optional: Save to JSON file
    output_file = f"round_{ROUND_TO_ANALYZE}_probabilities.json"
    with open(output_file, 'w') as f:
        json.dump(tick_probs, f, indent=2)
    print(f"\n💾 Full results saved to: {output_file}")
    
    # Create and display the plot
    print(f"\n📈 Creating visualization...")
    plot_save_path = VISUALIZATIONS_DIR / f"round_{ROUND_TO_ANALYZE}_probabilities.png"
    significant_changes = plot_round_probabilities(tick_probs, ROUND_TO_ANALYZE, save_path=plot_save_path)
    
    # Display significant probability changes table
    if significant_changes:
        print(f"\n⚡ Significant Probability Changes (≥2% delta):\n")
        print("=" * 110)
        print(f"{'Time':<8} {'Tick':<10} {'Delta':<10} {'Before':<12} {'After':<12} {'State':<10} {'Changes':<40}")
        print("=" * 110)
        
        for change in significant_changes:
            print(f"{change['time']:<8.1f} {change['tick']:<10} {change['delta']:<10.1f} "
                  f"{change['prob_before']:<12.1f} {change['prob_after']:<12.1f} "
                  f"{change['state']:<10} {change['changes']:<40}")
        
        print("=" * 110)
        print(f"\nTotal significant changes detected: {len(significant_changes)}")
    else:
        print("\nℹ️ No significant probability changes (≥2%) detected in this round.")


if __name__ == "__main__":
    main()
