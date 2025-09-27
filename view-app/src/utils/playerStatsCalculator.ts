/**
 * Player stats calculator for CS2 win probability prediction
 * Automatically calculates team-level stats and equipment from individual player features
 */

import type { FeatureValues } from '../types';

// Constants for weapon tiers
const MAIN_WEAPON_TIER_THRESHOLD = 5;

export interface PlayerData {
  side: number; // 0 = CT, 1 = T, -1 = Dead
  health: number;
  armor: number;
  has_helmet: boolean;
  has_defuser: boolean;
  best_weapon_tier: number;
}

export interface TeamStats {
  ct_main_weapons: number;
  t_main_weapons: number;
  ct_helmets: number;
  t_helmets: number;
  ct_armor: number;
  t_armor: number;
  defusers: number;
  ct_smokes: number;
  ct_flashes: number;
  ct_he_nades: number;
  ct_molotovs: number;
  t_smokes: number;
  t_flashes: number;
  t_he_nades: number;
  t_molotovs: number;
  cts_alive: number;
  ts_alive: number;
  hp_ct: number;
  hp_t: number;
}

/**
 * Extract individual player data from feature values
 */
export function extractPlayerData(featureValues: FeatureValues): PlayerData[] {
  const players: PlayerData[] = [];

  for (let i = 0; i < 10; i++) {
    const prefix = `player_${i}_`;

    // Check if this player has any data (if health is 0 or undefined, consider dead/inactive)
    const health = featureValues[`${prefix}health`] || 0;

    if (health > 0) {
      const player: PlayerData = {
        side: featureValues[`${prefix}side`] || 0,
        health: health,
        armor: featureValues[`${prefix}armor`] || 0,
        has_helmet: Boolean(featureValues[`${prefix}has_helmet`]),
        has_defuser: Boolean(featureValues[`${prefix}has_defuser`]),
        best_weapon_tier: featureValues[`${prefix}best_weapon_tier`] || 0,
      };

      players.push(player);
    }
  }

  return players;
}

/**
 * Calculate team stats from individual player data
 * Replicates the logic from the Python calculate_player_stats function
 */
export function calculateTeamStats(players: PlayerData[]): TeamStats {
  const stats: TeamStats = {
    ct_main_weapons: 0,
    t_main_weapons: 0,
    ct_helmets: 0,
    t_helmets: 0,
    ct_armor: 0,
    t_armor: 0,
    defusers: 0,
    ct_smokes: 0,
    ct_flashes: 0,
    ct_he_nades: 0,
    ct_molotovs: 0,
    t_smokes: 0,
    t_flashes: 0,
    t_he_nades: 0,
    t_molotovs: 0,
    cts_alive: 0,
    ts_alive: 0,
    hp_ct: 0,
    hp_t: 0,
  };

  for (const player of players) {
    const isCT = player.side === 0;
    const isT = player.side === 1;
    const hasArmor = player.armor > 0;

    // Count alive players
    if (isCT) {
      stats.cts_alive++;
      stats.hp_ct += player.health;
    } else if (isT) {
      stats.ts_alive++;
      stats.hp_t += player.health;
    }

    // Count main weapons (tier 5 and above)
    if (player.best_weapon_tier >= MAIN_WEAPON_TIER_THRESHOLD) {
      if (isCT) {
        stats.ct_main_weapons++;
      } else {
        stats.t_main_weapons++;
      }
    }

    // Count equipment
    if (isCT) {
      if (player.has_helmet) stats.ct_helmets++;
      if (hasArmor) stats.ct_armor++;
      // Note: Individual player features don't track specific grenades
      // We would need inventory data for accurate grenade counts
      // For now, we'll calculate based on reasonable defaults if needed
    } else {
      if (player.has_helmet) stats.t_helmets++;
      if (hasArmor) stats.t_armor++;
    }

    // Count defusers (only CTs can have defusers)
    if (player.has_defuser) {
      stats.defusers++;
    }
  }

  return stats;
}

/**
 * Get the list of feature names that should be automatically calculated from player data
 */
export function getCalculatedFeatureNames(): string[] {
  return [
    'ct_main_weapons',
    't_main_weapons',
    'ct_helmets',
    't_helmets',
    'ct_armor',
    't_armor',
    'defusers',
    'cts_alive',
    'ts_alive',
    'hp_ct',
    'hp_t',
    // Note: Grenade counts are not calculated as individual player features
    // don't track specific inventory items in the current data structure
  ];
}

/**
 * Check if a feature name represents an individual player feature
 */
export function isPlayerFeature(featureName: string): boolean {
  return featureName.startsWith('player_') && /^player_\d+_/.test(featureName);
}

/**
 * Check if a feature name represents a calculated team stat
 */
export function isCalculatedTeamStat(featureName: string): boolean {
  const calculatedFeatures = getCalculatedFeatureNames();
  return calculatedFeatures.includes(featureName);
}

/**
 * Main function to update calculated team stats based on player data
 */
export function applyContraints(featureValues: FeatureValues): FeatureValues {
  const updatedValues = { ...featureValues };

  Object.entries(featureValues).forEach(([statName, value]) => {
    if (statName === 'bomb_planted') {
      if (value) {
        updatedValues['round_time_left'] = 0;
      } else {
        updatedValues['bomb_time_left'] = 0;
      }
    }

    if (statName === 'cts_alive') {
      updatedValues['hp_ct'] = Math.min(value * 100, updatedValues['hp_ct']);
      updatedValues['ct_armor'] = Math.min(value, updatedValues['ct_armor']);
      updatedValues['ct_helmets'] = Math.min(
        value,
        updatedValues['ct_helmets']
      );
      updatedValues['ct_main_weapons'] = Math.min(
        value,
        updatedValues['ct_main_weapons']
      );
    }
    if (statName === 'ts_alive') {
      updatedValues['hp_t'] = Math.min(value * 100, updatedValues['hp_t']);
      updatedValues['t_armor'] = Math.min(value, updatedValues['t_armor']);
      updatedValues['t_helmets'] = Math.min(value, updatedValues['t_helmets']);
      updatedValues['t_main_weapons'] = Math.min(
        value,
        updatedValues['t_main_weapons']
      );
    }
  });

  const players = extractPlayerData(updatedValues);

  if (players.length === 0) return updatedValues;

  const teamStats = calculateTeamStats(players);

  // Create a copy of feature values and update calculated stats

  // Update all calculated team stats
  Object.entries(teamStats).forEach(([statName, value]) => {
    if (getCalculatedFeatureNames().includes(statName)) {
      updatedValues[statName] = value;
    }
  });

  return updatedValues;
}

/**
 * Estimate reasonable grenade counts based on team composition and economy
 * This is a heuristic since individual player features don't track specific inventory
 */
export function estimateGrenadeStats(
  players: PlayerData[]
): Partial<TeamStats> {
  const grenadeStats: Partial<TeamStats> = {};

  // Simple heuristic: assume players with good weapons have more utility
  const ctPlayersWithGoodWeapons = players.filter(
    (p) => p.side === 0 && p.best_weapon_tier >= 5
  ).length;
  const tPlayersWithGoodWeapons = players.filter(
    (p) => p.side === 1 && p.best_weapon_tier >= 5
  ).length;

  // Estimate based on weapon economy (better weapons = more utility)
  grenadeStats.ct_smokes = Math.min(
    Math.floor(ctPlayersWithGoodWeapons * 0.4),
    2
  );
  grenadeStats.ct_flashes = Math.min(
    Math.floor(ctPlayersWithGoodWeapons * 0.6),
    3
  );
  grenadeStats.ct_he_nades = Math.min(
    Math.floor(ctPlayersWithGoodWeapons * 0.3),
    2
  );
  grenadeStats.ct_molotovs = Math.min(
    Math.floor(ctPlayersWithGoodWeapons * 0.2),
    1
  );

  grenadeStats.t_smokes = Math.min(
    Math.floor(tPlayersWithGoodWeapons * 0.5),
    2
  );
  grenadeStats.t_flashes = Math.min(
    Math.floor(tPlayersWithGoodWeapons * 0.7),
    4
  );
  grenadeStats.t_he_nades = Math.min(
    Math.floor(tPlayersWithGoodWeapons * 0.4),
    2
  );
  grenadeStats.t_molotovs = Math.min(
    Math.floor(tPlayersWithGoodWeapons * 0.3),
    1
  );

  return grenadeStats;
}
