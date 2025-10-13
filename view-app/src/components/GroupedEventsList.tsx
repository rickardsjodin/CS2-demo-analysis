import { useMemo, useState } from 'react';
import type { PlayerAnalysisEvent } from '../types';

interface GroupedEventsListProps {
  data: PlayerAnalysisEvent[];
  playerName: string;
}

interface RoundGroup {
  round: number;
  events: PlayerAnalysisEvent[];
  totalImpact: number;
}

function GroupedEventsList({ data, playerName }: GroupedEventsListProps) {
  const [copySuccess, setCopySuccess] = useState(false);

  // Group events by side and round
  const groupedEvents = useMemo(() => {
    const ctEvents = data.filter((e) => e.side === 'ct');
    const tEvents = data.filter((e) => e.side === 't');

    // Group by round
    const ctByRound = new Map<number, PlayerAnalysisEvent[]>();
    const tByRound = new Map<number, PlayerAnalysisEvent[]>();

    ctEvents.forEach((e) => {
      if (!ctByRound.has(e.round)) {
        ctByRound.set(e.round, []);
      }
      ctByRound.get(e.round)!.push(e);
    });

    tEvents.forEach((e) => {
      if (!tByRound.has(e.round)) {
        tByRound.set(e.round, []);
      }
      tByRound.get(e.round)!.push(e);
    });

    // Convert to sorted arrays
    const ctRounds = Array.from(ctByRound.entries())
      .sort(([a], [b]) => a - b)
      .map(([round, events]) => ({
        round,
        events,
        totalImpact: events.reduce((sum, e) => sum + e.impact, 0),
      }));

    const tRounds = Array.from(tByRound.entries())
      .sort(([a], [b]) => a - b)
      .map(([round, events]) => ({
        round,
        events,
        totalImpact: events.reduce((sum, e) => sum + e.impact, 0),
      }));

    return { ctRounds, tRounds };
  }, [data]);

  // Generate copyable text
  const generateCopyText = () => {
    let text = `${playerName} - Impact Events Summary\n\n`;

    text += `=== CT SIDE ===\n\n`;
    groupedEvents.ctRounds.forEach(({ round, events, totalImpact }) => {
      text += `Round ${round} (Delta: ${
        totalImpact >= 0 ? '+' : ''
      }${totalImpact.toFixed(1)}):\n`;
      events.forEach((e) => {
        text += `  ${e.impact >= 0 ? '+' : ''}${e.impact.toFixed(1)} | ${
          e.game_state
        } | ${(e.pre_win * 100).toFixed(1)}% → ${(e.post_win * 100).toFixed(
          1
        )}%\n`;
      });
      text += '\n';
    });

    text += `=== T SIDE ===\n\n`;
    groupedEvents.tRounds.forEach(({ round, events, totalImpact }) => {
      text += `Round ${round} (Delta: ${
        totalImpact >= 0 ? '+' : ''
      }${totalImpact.toFixed(1)}):\n`;
      events.forEach((e) => {
        text += `  ${e.impact >= 0 ? '+' : ''}${e.impact.toFixed(1)} | ${
          e.game_state
        } | ${(e.pre_win * 100).toFixed(1)}% → ${(e.post_win * 100).toFixed(
          1
        )}%\n`;
      });
      text += '\n';
    });

    return text;
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(generateCopyText());
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const renderRoundGroup = (roundGroup: RoundGroup) => {
    const { round, events, totalImpact } = roundGroup;
    return (
      <div
        key={round}
        style={{
          marginBottom: '15px',
          padding: '10px',
          backgroundColor: '#1e293b',
          borderRadius: '4px',
        }}
      >
        <div
          style={{
            fontWeight: 'bold',
            marginBottom: '8px',
            color: totalImpact >= 0 ? '#4caf50' : '#f44336',
          }}
        >
          Round {round}: {totalImpact >= 0 ? '+' : ''}
          {totalImpact.toFixed(1)}
        </div>
        {events.map((e, idx) => (
          <div
            key={idx}
            style={{
              padding: '4px 0 4px 12px',
              fontSize: '13px',
              borderLeft: `3px solid ${e.impact >= 0 ? '#4caf50' : '#f44336'}`,
            }}
          >
            <span
              style={{
                fontWeight: 'bold',
                color: e.impact >= 0 ? '#4caf50' : '#f44336',
              }}
            >
              {e.impact >= 0 ? '+' : ''}
              {e.impact.toFixed(1)}
            </span>
            {' | '}
            <span>{e.game_state}</span>
            {' | '}
            <span style={{ fontSize: '11px', color: '#94a3b8' }}>
              {(e.pre_win * 100).toFixed(1)}% → {(e.post_win * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className='events-table-container'>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '15px',
        }}
      >
        <h3>Grouped Impact Events</h3>
        <button
          onClick={handleCopy}
          style={{
            padding: '8px 16px',
            backgroundColor: copySuccess ? '#4caf50' : '#2196f3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '500',
            transition: 'all 0.3s ease',
          }}
        >
          {copySuccess ? '✓ Copied!' : '📋 Copy Summary'}
        </button>
      </div>

      <div
        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}
      >
        {/* CT Side */}
        <div>
          <h4 style={{ color: '#2196f3', marginBottom: '10px' }}>
            CT SIDE ({groupedEvents.ctRounds.length} rounds)
          </h4>

          <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
            {groupedEvents.ctRounds.map(renderRoundGroup)}
          </div>
        </div>

        {/* T Side */}
        <div>
          <h4 style={{ color: '#ff9800', marginBottom: '10px' }}>
            T SIDE ({groupedEvents.tRounds.length} rounds)
          </h4>

          <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
            {groupedEvents.tRounds.map(renderRoundGroup)}
          </div>
        </div>
      </div>
    </div>
  );
}

export default GroupedEventsList;
