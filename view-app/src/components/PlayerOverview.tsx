import { useMemo } from 'react';
import type { PlayerAnalysisEvent } from '../types';
import './PlayerOverview.css';

interface PlayerOverviewProps {
  allPlayersData: Record<string, PlayerAnalysisEvent[]> | null;
  selectedPlayer: string | null;
  onPlayerSelect: (playerName: string) => void;
  analyzing: boolean;
}

interface PlayerSummary {
  name: string;
  totalImpact: number;
  eventCount: number;
  roundCount: number;
  avgImpact: number;
  side: 'ct' | 't' | 'mixed';
}

function PlayerOverview({
  allPlayersData,
  selectedPlayer,
  onPlayerSelect,
  analyzing,
}: PlayerOverviewProps) {
  // Calculate player summaries and group by team
  const teamGroups = useMemo(() => {
    if (!allPlayersData)
      return {
        team1: [],
        team2: [],
        team1Total: 0,
        team2Total: 0,
      };

    const playerSummaries: PlayerSummary[] = [];

    for (const [playerName, events] of Object.entries(allPlayersData)) {
      const totalImpact = events.reduce((sum, event) => sum + event.impact, 0);

      // Count unique rounds the player participated in
      const uniqueRounds = new Set(events.map((event) => event.round));
      const roundCount = uniqueRounds.size;
      const avgImpact = roundCount > 0 ? totalImpact / roundCount : 0;

      // Determine player's primary side (the side they played most rounds on)
      const sideCounts = { ct: 0, t: 0 };
      events.slice(0, 13).forEach((event) => {
        sideCounts[event.side]++;
      });

      let side: 'ct' | 't' | 'mixed';
      if (sideCounts.ct > sideCounts.t * 1.5) {
        side = 'ct';
      } else if (sideCounts.t > sideCounts.ct * 1.5) {
        side = 't';
      } else {
        side = 'mixed';
      }

      playerSummaries.push({
        name: playerName,
        totalImpact,
        eventCount: events.length,
        roundCount,
        avgImpact,
        side,
      });
    }

    // Group by team and sort by average impact per round
    const team1Players = playerSummaries
      .filter((p) => p.side === 'ct' || p.side === 'mixed')
      .sort((a, b) => b.avgImpact - a.avgImpact);

    const team2Players = playerSummaries
      .filter((p) => p.side === 't' || p.side === 'mixed')
      .sort((a, b) => b.avgImpact - a.avgImpact);

    // Calculate total average impact for each team
    const team1Total = team1Players.reduce((sum, p) => sum + p.avgImpact, 0);
    const team2Total = team2Players.reduce((sum, p) => sum + p.avgImpact, 0);

    return {
      team1: team1Players,
      team2: team2Players,
      team1Total,
      team2Total,
    };
  }, [allPlayersData]);

  if (!allPlayersData) return null;

  return (
    <section className='player-overview-section'>
      <h2>🎮 Players Impact Overview</h2>

      <div className='teams-container'>
        {/* Team 1 */}
        <div className='team-group team1'>
          <h3 className='team-header team1-header'>
            <span className='team-badge team1'>Team 1</span>
            <span className='team-count'>
              {teamGroups.team1.length} players
            </span>
            <span
              className={`team-total ${
                teamGroups.team1Total >= 0 ? 'positive' : 'negative'
              }`}
            >
              Total: {teamGroups.team1Total >= 0 ? '+' : ''}
              {teamGroups.team1Total.toFixed(1)}%
            </span>
          </h3>
          <div className='players-list'>
            {teamGroups.team1.map((player) => (
              <button
                key={player.name}
                className={`player-card ${
                  selectedPlayer === player.name ? 'selected' : ''
                } ${player.avgImpact >= 0 ? 'positive' : 'negative'}`}
                onClick={() => onPlayerSelect(player.name)}
                disabled={analyzing}
              >
                <div className='player-name'>{player.name}</div>
                <div className='player-stats'>
                  <span
                    className={`impact-value ${
                      player.avgImpact >= 0 ? 'positive' : 'negative'
                    }`}
                  >
                    {player.avgImpact >= 0 ? '+' : ''}
                    {player.avgImpact.toFixed(1)}%
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Team 2 */}
        <div className='team-group team2'>
          <h3 className='team-header team2-header'>
            <span className='team-badge team2'>Team 2</span>
            <span className='team-count'>
              {teamGroups.team2.length} players
            </span>
            <span
              className={`team-total ${
                teamGroups.team2Total >= 0 ? 'positive' : 'negative'
              }`}
            >
              Total: {teamGroups.team2Total >= 0 ? '+' : ''}
              {teamGroups.team2Total.toFixed(1)}%
            </span>
          </h3>
          <div className='players-list'>
            {teamGroups.team2.map((player) => (
              <button
                key={player.name}
                className={`player-card ${
                  selectedPlayer === player.name ? 'selected' : ''
                } ${player.avgImpact >= 0 ? 'positive' : 'negative'}`}
                onClick={() => onPlayerSelect(player.name)}
                disabled={analyzing}
              >
                <div className='player-name'>{player.name}</div>
                <div className='player-stats'>
                  <span
                    className={`impact-value ${
                      player.avgImpact >= 0 ? 'positive' : 'negative'
                    }`}
                  >
                    {player.avgImpact >= 0 ? '+' : ''}
                    {player.avgImpact.toFixed(1)}%
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export default PlayerOverview;
