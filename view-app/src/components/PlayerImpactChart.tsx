import { useMemo } from 'react';
import type { PlayerAnalysisEvent } from '../types';
import ImpactCanvasChart from './ImpactCanvasChart';
import GroupedEventsList from './GroupedEventsList';
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
    // Calculate totals from individual events (not aggregated rounds)
    const totalImpact = data.reduce((sum, e) => sum + e.impact, 0);
    const positiveImpact = data.reduce(
      (sum, e) => sum + (e.impact > 0 ? e.impact : 0),
      0
    );
    const negativeImpact = data.reduce(
      (sum, e) => sum + (e.impact < 0 ? e.impact : 0),
      0
    );
    const avgImpact = totalImpact / roundData.length;

    // Calculate per-side statistics using original event data
    const ctEvents = data.filter((e) => e.side === 'ct');
    const tEvents = data.filter((e) => e.side === 't');

    // Per-side totals (from original events, not aggregated rounds)
    const ctTotalImpact = ctEvents.reduce((sum, e) => sum + e.impact, 0);
    const tTotalImpact = tEvents.reduce((sum, e) => sum + e.impact, 0);

    // Per-side positive/negative impacts (from original events)
    const ctPositiveImpact = ctEvents.reduce(
      (sum, e) => sum + (e.impact > 0 ? e.impact : 0),
      0
    );
    const ctNegativeImpact = ctEvents.reduce(
      (sum, e) => sum + (e.impact < 0 ? e.impact : 0),
      0
    );
    const tPositiveImpact = tEvents.reduce(
      (sum, e) => sum + (e.impact > 0 ? e.impact : 0),
      0
    );
    const tNegativeImpact = tEvents.reduce(
      (sum, e) => sum + (e.impact < 0 ? e.impact : 0),
      0
    );

    // Count rounds per side
    const ctRounds = roundData.filter((r) => r.side === 'ct');
    const tRounds = roundData.filter((r) => r.side === 't');

    const ctAvgImpact =
      ctRounds.length > 0 ? ctTotalImpact / ctRounds.length : 0;
    const tAvgImpact = tRounds.length > 0 ? tTotalImpact / tRounds.length : 0;

    return {
      totalImpact: totalImpact.toFixed(1),
      avgImpact: avgImpact.toFixed(1),
      positiveImpact: positiveImpact.toFixed(1),
      negativeImpact: negativeImpact.toFixed(1),
      rounds: roundData.length,
      ctAvgImpact: ctAvgImpact.toFixed(1),
      tAvgImpact: tAvgImpact.toFixed(1),
      ctRounds: ctRounds.length,
      tRounds: tRounds.length,
      ctPositiveImpact: ctPositiveImpact.toFixed(1),
      ctNegativeImpact: ctNegativeImpact.toFixed(1),
      tPositiveImpact: tPositiveImpact.toFixed(1),
      tNegativeImpact: tNegativeImpact.toFixed(1),
    };
  }, [roundData, data]);

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

      {/* Per-Side Statistics */}
      <div className='stats-summary'>
        <div className='stat-card ct-side-card'>
          <div className='stat-label'>
            CT Avg Impact ({stats.ctRounds} rounds)
          </div>
          <div className='stat-value'>{stats.ctAvgImpact}</div>
        </div>
        <div className='stat-card t-side-card'>
          <div className='stat-label'>
            T Avg Impact ({stats.tRounds} rounds)
          </div>
          <div className='stat-value'>{stats.tAvgImpact}</div>
        </div>
      </div>

      {/* Per-Side Positive/Negative Impact */}
      <div className='stats-summary'>
        <div className='stat-card ct-side-card positive'>
          <div className='stat-label'>CT Positive Impact</div>
          <div className='stat-value'>+{stats.ctPositiveImpact}</div>
        </div>
        <div className='stat-card ct-side-card negative'>
          <div className='stat-label'>CT Negative Impact</div>
          <div className='stat-value'>{stats.ctNegativeImpact}</div>
        </div>
        <div className='stat-card t-side-card positive'>
          <div className='stat-label'>T Positive Impact</div>
          <div className='stat-value'>+{stats.tPositiveImpact}</div>
        </div>
        <div className='stat-card t-side-card negative'>
          <div className='stat-label'>T Negative Impact</div>
          <div className='stat-value'>{stats.tNegativeImpact}</div>
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

      {/* Grouped Events List */}
      <GroupedEventsList data={data} playerName={playerName} />
    </div>
  );
}

export default PlayerImpactChart;
