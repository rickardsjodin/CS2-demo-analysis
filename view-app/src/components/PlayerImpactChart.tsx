import { useMemo } from 'react';
import type { PlayerAnalysisEvent } from '../types';
import ImpactCanvasChart from './ImpactCanvasChart';
import './PlayerImpactChart.css';

interface PlayerImpactChartProps {
  data: PlayerAnalysisEvent[];
  playerName: string;
}

function PlayerImpactChart({ data, playerName }: PlayerImpactChartProps) {
  // Aggregate impact by round
  const roundData = useMemo(() => {
    const rounds = new Map<
      number,
      { impact: number; side: string; events: number }
    >();

    data.forEach((event) => {
      const existing = rounds.get(event.round);
      if (existing) {
        existing.impact += event.impact;
        existing.events += 1;
      } else {
        rounds.set(event.round, {
          impact: event.impact,
          side: event.side,
          events: 1,
        });
      }
    });

    return Array.from(rounds.entries())
      .map(([round, stats]) => ({
        round,
        ...stats,
      }))
      .sort((a, b) => a.round - b.round);
  }, [data]);

  // Calculate statistics
  const stats = useMemo(() => {
    const totalImpact = roundData.reduce((sum, r) => sum + r.impact, 0);
    const positiveImpact = roundData.reduce(
      (sum, r) => sum + (r.impact > 0 ? r.impact : 0),
      0
    );
    const negativeImpact = roundData.reduce(
      (sum, r) => sum + (r.impact < 0 ? r.impact : 0),
      0
    );
    const avgImpact = totalImpact / roundData.length;

    return {
      totalImpact: totalImpact.toFixed(1),
      avgImpact: avgImpact.toFixed(1),
      positiveImpact: positiveImpact.toFixed(1),
      negativeImpact: negativeImpact.toFixed(1),
      rounds: roundData.length,
    };
  }, [roundData]);

  // Find max absolute impact for scaling
  const maxImpact = useMemo(
    () => Math.max(...roundData.map((r) => Math.abs(r.impact))),
    [roundData]
  );

  return (
    <div className='player-impact-chart'>
      {/* Statistics Summary */}
      <div className='stats-summary'>
        <div className='stat-card'>
          <div className='stat-label'>Total Impact</div>
          <div className='stat-value'>{stats.totalImpact}</div>
        </div>
        <div className='stat-card'>
          <div className='stat-label'>Avg Impact/Round</div>
          <div className='stat-value'>{stats.avgImpact}</div>
        </div>
        <div className='stat-card positive'>
          <div className='stat-label'>Positive Impact</div>
          <div className='stat-value'>+{stats.positiveImpact}</div>
        </div>
        <div className='stat-card negative'>
          <div className='stat-label'>Negative Impact</div>
          <div className='stat-value'>{stats.negativeImpact}</div>
        </div>
      </div>

      {/* Canvas-based Individual Impact Chart */}
      <div className='canvas-chart-container'>
        <h3>Individual Impact Events by Round</h3>
        <ImpactCanvasChart
          data={data}
          playerName={playerName}
          width={1200}
          height={500}
        />
      </div>

      {/* Bar Chart - Aggregated by Round */}
      <div className='chart-container'>
        <h3>Aggregated Impact by Round</h3>
        <div className='chart'>
          {roundData.map((round) => {
            const heightPercent = (Math.abs(round.impact) / maxImpact) * 100;
            const isPositive = round.impact >= 0;

            return (
              <div key={round.round} className='chart-bar-wrapper'>
                <div
                  className={`chart-bar ${
                    isPositive ? 'positive' : 'negative'
                  } ${round.side === 'ct' ? 'ct-side' : 't-side'}`}
                  style={{
                    height: `${Math.max(heightPercent, 2)}%`,
                  }}
                  title={`Round ${round.round}: ${round.impact.toFixed(
                    1
                  )} (${round.side.toUpperCase()})`}
                >
                  <span className='bar-value'>{round.impact.toFixed(1)}</span>
                </div>
                <div className='chart-label'>{round.round}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Event Details Table */}
      <div className='events-table-container'>
        <h3>Event Details</h3>
        <table className='events-table'>
          <thead>
            <tr>
              <th>Round</th>
              <th>Side</th>
              <th>Game State</th>
              <th>Impact</th>
              <th>Pre Win %</th>
              <th>Post Win %</th>
              <th>Post Plant</th>
            </tr>
          </thead>
          <tbody>
            {data.map((event, idx) => (
              <tr key={idx}>
                <td>{event.round}</td>
                <td>
                  <span className={`side-badge ${event.side}`}>
                    {event.side.toUpperCase()}
                  </span>
                </td>
                <td>{event.game_state}</td>
                <td
                  className={
                    event.impact >= 0 ? 'positive-impact' : 'negative-impact'
                  }
                >
                  {event.impact >= 0 ? '+' : ''}
                  {event.impact.toFixed(1)}
                </td>
                <td>{(event.pre_win * 100).toFixed(1)}%</td>
                <td>{(event.post_win * 100).toFixed(1)}%</td>
                <td>{event.post_plant ? '💣' : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default PlayerImpactChart;
