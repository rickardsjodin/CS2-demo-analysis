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


def plot_round_probabilities(tick_probs_list, round_num, model_names=None, save_path=None):
    """
    Create a visualization of probability changes throughout a round.
    
    Args:
        tick_probs_list: List of tick probability dictionaries OR list of such lists (for multiple models)
        round_num: Round number being analyzed
        model_names: List of model names (required if tick_probs_list contains multiple models)
        save_path: Optional path to save the plot
    """
    # Handle both single model and multiple models
    if not tick_probs_list:
        print("No data to plot")
        return
    
    # Check if this is a single model (list of dicts) or multiple models (list of lists)
    is_multi_model = isinstance(tick_probs_list[0], list)
    
    if not is_multi_model:
        # Convert single model to list format for consistent processing
        tick_probs_list = [tick_probs_list]
        model_names = model_names or ['Model']
    
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(tick_probs_list))]
    
    if len(model_names) != len(tick_probs_list):
        print(f"Warning: Number of model names ({len(model_names)}) doesn't match number of models ({len(tick_probs_list)})")
        model_names = [f'Model {i+1}' for i in range(len(tick_probs_list))]
    
    # Use the first model for reference data (kills, bomb plants, etc.)
    tick_probs = tick_probs_list[0]
    
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
    
    # Define colors for different models
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    # Plot each model's CT probability line
    for idx, (tick_probs_model, model_name) in enumerate(zip(tick_probs_list, model_names)):
        time_elapsed_model = [p['time_elapsed'] for p in tick_probs_model]
        ct_prob_model = [p['ct_win_probability'] * 100 for p in tick_probs_model]
        
        color = colors[idx % len(colors)]
        ax.plot(time_elapsed_model, ct_prob_model, label=f'{model_name} - CT Win', 
                color=color, linewidth=2.5, alpha=0.8)
        
        # Only fill for the first model to avoid clutter
        if idx == 0:
            ax.fill_between(time_elapsed_model, ct_prob_model, 50, 
                           where=[p >= 50 for p in ct_prob_model],
                           color=color, alpha=0.1, interpolate=True, label='CT Favored')
            ax.fill_between(time_elapsed_model, ct_prob_model, 50, 
                           where=[p < 50 for p in ct_prob_model],
                           color='#e74c3c', alpha=0.1, interpolate=True, label='T Favored')
    
    # Mark kills with vertical lines only (no labels, since annotations handle this)
    for kt in kill_times:
        ax.axvline(x=kt, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Mark significant probability changes (>=1%) with annotations showing the reason
    for idx, change in enumerate(significant_changes):
        change_time = change['time']
        change_prob = change['prob_after']
        change_reason = change['changes']
        
        # Add annotation with the actual reason for the change
        # Alternate annotation positions to avoid overlap
        y_offset = 12 if idx % 2 == 0 else -12
        va = 'bottom' if idx % 2 == 0 else 'top'
        
        ax.annotate(change_reason, 
                   xy=(change_time, change_prob), 
                   xytext=(change_time, change_prob + y_offset),
                   ha='center',
                   va=va,
                   fontsize=8,
                   color='#9b59b6',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='#9b59b6', linewidth=1.5, alpha=0.85),
                   arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))
    
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
    DEMO_FILE = "C:\\Users\\Rickard\\Downloads\\esl-pro-league-season-19-forze-vs-furia-bo3-peZBbn8_BvGatAm4iQTGtP\\forze-vs-furia-m2-overpass.dem"
    ROUND_TO_ANALYZE = 13  # Change this to analyze different rounds
    TICK_STEP = 10  # Sample every Nth tick (default: 10, means ~6.4 samples per second at 64 tick rate)
    
    # Define multiple models to compare
    MODEL_FILES = [
        MODELS_DIR / "ct_win_probability_xgboost_minimal.pkl",
        # Add more model paths here to compare them
        MODELS_DIR / "ct_win_probability_xgboost_hltv.pkl",
        # MODELS_DIR / "ct_win_probability_lightgbm.pkl",
    ]
    
    MODEL_NAMES = [
        "XGBoost Minimal",
        # Add corresponding names here
        "XGBoost HLTV",
        # "LightGBM",
    ]
    
    # Load the demo
    print(f"📂 Loading demo: {DEMO_FILE}")
    dem = load_demo(str(DEMO_FILE), use_cache=True)
    
    if dem is None:
        print("❌ Failed to load demo")
        return
    
    print("✅ Demo loaded successfully!")
    
    # Load the prediction models
    pred_models = []
    for i, model_file in enumerate(MODEL_FILES):
        print(f"🤖 Loading model {i+1}/{len(MODEL_FILES)}: {model_file}")
        try:
            with open(model_file, 'rb') as f:
                pred_model = pickle.load(f)
            print(f"✅ Model loaded successfully!")
            pred_models.append(pred_model)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
    
    # Calculate probabilities for every tick with each model
    all_tick_probs = []
    for i, (pred_model, model_name) in enumerate(zip(pred_models, MODEL_NAMES)):
        print(f"\n🔍 Analyzing Round {ROUND_TO_ANALYZE} with {model_name} (tick step: {TICK_STEP})...")
        tick_probs = calculate_round_probabilities_all_ticks(
            dem_file=dem,
            round_num=ROUND_TO_ANALYZE,
            pred_model=pred_model,
            tick_rate=64,
            tick_step=TICK_STEP,
            file_name=DEMO_FILE
        )
        
        if not tick_probs:
            print(f"❌ No probabilities calculated for round {ROUND_TO_ANALYZE} with {model_name}")
            return
        
        all_tick_probs.append(tick_probs)
        print(f"✅ Calculated {len(tick_probs)} ticks for {model_name}")
    
    # Use first model's results for display
    tick_probs = all_tick_probs[0]
    
    if not tick_probs:
        print(f"❌ No probabilities calculated for round {ROUND_TO_ANALYZE}")
        return
    
    # Display results
    print(f"\n✅ Calculated probabilities for {len(tick_probs)} ticks\n")
    
    # Show comparison table for multiple models at key intervals
    if len(all_tick_probs) > 1:
        print("\n" + "=" * 120)
        print(f"📊 Model Comparison at Key Time Points")
        print("=" * 120)
        header = f"{'Tick':<8} {'Time (s)':<10} {'State':<10} {'Bomb':<8}"
        for model_name in MODEL_NAMES:
            header += f" {model_name[:15]:<15}"
        print(header)
        print("=" * 120)
        
        # Show every 64 ticks (1 second) or at events
        sample_indices = list(range(0, len(tick_probs), 64)) + [len(tick_probs) - 1]
        sample_indices = sorted(set(sample_indices))
        
        for idx in sample_indices:
            if idx >= len(tick_probs):
                continue
            entry = tick_probs[idx]
            state = f"{entry['cts_alive']}v{entry['ts_alive']}"
            bomb_status = "Yes" if entry['bomb_planted'] else "No"
            
            row = f"{entry['tick']:<8} {entry['time_elapsed']:<10.2f} {state:<10} {bomb_status:<8}"
            for model_probs in all_tick_probs:
                if idx < len(model_probs):
                    ct_prob = model_probs[idx]['ct_win_probability'] * 100
                    row += f" {ct_prob:<15.1f}"
                else:
                    row += f" {'N/A':<15}"
            print(row)
        
        print("=" * 120)
    
    # Show sample of results (first 10, every 64th tick, and last 10) for first model
    print("\n" + "=" * 100)
    print(f"Detailed Results - {MODEL_NAMES[0]}")
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
    
    # Summary statistics for all models
    print(f"\n📊 Summary Statistics:")
    for i, (tick_probs_model, model_name) in enumerate(zip(all_tick_probs, MODEL_NAMES)):
        ct_probs = [p['ct_win_probability'] for p in tick_probs_model]
        print(f"\n  {model_name}:")
        print(f"     Average CT Win Probability: {sum(ct_probs) / len(ct_probs) * 100:.1f}%")
        print(f"     Max CT Win Probability: {max(ct_probs) * 100:.1f}%")
        print(f"     Min CT Win Probability: {min(ct_probs) * 100:.1f}%")
        print(f"     Total ticks analyzed: {len(tick_probs_model)}")
        if i == 0:
            print(f"     Round duration: {tick_probs_model[-1]['time_elapsed']:.1f} seconds")
    
    # Optional: Save to JSON file
    output_file = f"round_{ROUND_TO_ANALYZE}_probabilities.json"
    with open(output_file, 'w') as f:
        json.dump(tick_probs, f, indent=2)
    print(f"\n💾 Full results saved to: {output_file}")
    
    # Create and display the plot with all models
    print(f"\n📈 Creating visualization...")
    plot_save_path = VISUALIZATIONS_DIR / f"round_{ROUND_TO_ANALYZE}_probabilities_comparison.png"
    
    # Pass all models' data to the plotting function
    if len(all_tick_probs) > 1:
        significant_changes = plot_round_probabilities(all_tick_probs, ROUND_TO_ANALYZE, 
                                                      model_names=MODEL_NAMES, save_path=plot_save_path)
    else:
        significant_changes = plot_round_probabilities(tick_probs, ROUND_TO_ANALYZE, 
                                                      save_path=plot_save_path)
    
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
