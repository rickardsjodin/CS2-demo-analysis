"""
Win probability calculations for CS2 rounds based on player counts and game state.
"""

def get_win_probability(ct_alive, t_alive, post_plant=False):
    """
    Calculate CT win probability based on alive players and game state.
    
    Args:
        ct_alive: Number of CT players alive
        t_alive: Number of T players alive
        post_plant: Whether the bomb has been planted
    
    Returns:
        CT win probability (0.0 to 1.0)
    """
    if ct_alive == 0:
        return 0.0  # T wins, no CTs left
    if t_alive == 0:
        return 1.0  # CT wins
    
    # Probability lookup table based on provided data
    # Format: win_probabilities[t_alive][ct_alive] = T win probability
    win_probabilities = {
        5: {5: 0.488, 4: 0.702, 3: 0.890, 2: 0.981, 1: 0.999},
        4: {5: 0.315, 4: 0.525, 3: 0.769, 2: 0.942, 1: 0.996},
        3: {5: 0.143, 4: 0.300, 3: 0.546, 2: 0.819, 1: 0.974},
        2: {5: 0.032, 4: 0.099, 3: 0.258, 2: 0.547, 1: 0.870},
        1: {5: 0.002, 4: 0.009, 3: 0.045, 2: 0.189, 1: 0.551}
    }
    
    # Get base T win probability
    if t_alive in win_probabilities and ct_alive in win_probabilities[t_alive]:
        t_win_prob = win_probabilities[t_alive][ct_alive]
    else:
        # Fallback for edge cases
        ratio = ct_alive / (ct_alive + t_alive)
        t_win_prob = 1 - (ratio * 0.8 + 0.1)
    
    # Convert to CT win probability
    ct_win_prob = 1 - t_win_prob
    
    # Adjust for post-plant scenarios with more nuanced calculation
    if post_plant:
        # Base post-plant boost + scaling based on player imbalance
        player_ratio = t_alive / ct_alive
        post_plant_boost = 0.08 + (0.10 * player_ratio)
        
        # Cap the boost and apply it
        t_win_prob = min(0.90, t_win_prob + post_plant_boost)
        ct_win_prob = 1 - t_win_prob
    
    return ct_win_prob


def calculate_impact_score(ct_before, t_before, ct_after, t_after, is_post_plant=False):
    """
    Calculate the impact score of a kill/death event.
    
    Args:
        ct_before: CT players alive before the kill
        t_before: T players alive before the kill
        ct_after: CT players alive after the kill
        t_after: T players alive after the kill
        is_post_plant: Whether bomb is planted
    
    Returns:
        Impact score (higher = more impactful)
    """
    # Calculate win probabilities before and after
    prob_before = get_win_probability(ct_before, t_before, is_post_plant)
    prob_after = get_win_probability(ct_after, t_after, is_post_plant)
    
    # Impact is the change in win probability (0-100 scale)
    impact = abs(prob_after - prob_before) * 100
    
    return round(impact, 1)
