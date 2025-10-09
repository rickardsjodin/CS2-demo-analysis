import { useEffect, useRef, useState, useCallback } from 'react';
import type { PlayerAnalysisEvent } from '../types';

interface ImpactCanvasChartProps {
  data: PlayerAnalysisEvent[];
  playerName: string;
  width?: number;
  height?: number;
}

interface BarPosition {
  x: number;
  y: number;
  width: number;
  height: number;
  event: PlayerAnalysisEvent;
}

function ImpactCanvasChart({
  data,
  playerName,
  width = 1200,
  height = 500,
}: ImpactCanvasChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    event: PlayerAnalysisEvent;
  } | null>(null);
  const barPositionsRef = useRef<BarPosition[]>([]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Find if mouse is over any bar
      const hoveredBar = barPositionsRef.current.find(
        (bar) =>
          x >= bar.x &&
          x <= bar.x + bar.width &&
          y >= Math.min(bar.y, bar.y + bar.height) &&
          y <= Math.max(bar.y, bar.y + bar.height)
      );

      if (hoveredBar) {
        setTooltip({
          x: e.clientX,
          y: e.clientY,
          event: hoveredBar.event,
        });
      } else {
        setTooltip(null);
      }
    },
    []
  );

  const handleMouseLeave = useCallback(() => {
    setTooltip(null);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size accounting for device pixel ratio
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Organize data by round
    const roundMap = new Map<number, PlayerAnalysisEvent[]>();
    data.forEach((event) => {
      if (!roundMap.has(event.round)) {
        roundMap.set(event.round, []);
      }
      roundMap.get(event.round)!.push(event);
    });

    const rounds = Array.from(roundMap.keys()).sort((a, b) => a - b);

    // Chart dimensions
    const padding = { top: 60, right: 40, bottom: 80, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Find max absolute impact for scaling
    const allImpacts = data.map((e) => Math.abs(e.impact));
    const maxImpact = Math.max(...allImpacts, 30); // Minimum scale of 30

    // Calculate bar positions
    const totalBars = data.length;
    const barWidth = Math.max(
      15,
      Math.min(40, chartWidth / (totalBars + rounds.length * 2))
    );
    const roundGap = barWidth * 2;

    // Draw background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    const gridSteps = 5;
    for (let i = 0; i <= gridSteps; i++) {
      const y = padding.top + (chartHeight / gridSteps) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();

      // Grid labels
      const value = maxImpact - (maxImpact * 2 * i) / gridSteps;
      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(value.toFixed(0), padding.left - 10, y + 4);
    }

    // Draw zero line
    const zeroY = padding.top + chartHeight / 2;
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, zeroY);
    ctx.lineTo(padding.left + chartWidth, zeroY);
    ctx.stroke();

    // Draw bars by round
    let currentX = padding.left;
    barPositionsRef.current = [];

    rounds.forEach((roundNum, roundIndex) => {
      const events = roundMap.get(roundNum)!;
      const roundStartX = currentX;

      // Draw round background
      const roundWidth = events.length * barWidth + (events.length - 1) * 2;
      const side = events[0].side;
      ctx.fillStyle =
        side === 'ct' ? 'rgba(33, 150, 243, 0.05)' : 'rgba(255, 152, 0, 0.05)';
      ctx.fillRect(roundStartX, padding.top, roundWidth, chartHeight);

      // Draw individual event bars
      events.forEach((event) => {
        const barHeight =
          (Math.abs(event.impact) / maxImpact) * (chartHeight / 2);
        const barY = event.impact >= 0 ? zeroY - barHeight : zeroY;

        // Store bar position for hover detection
        barPositionsRef.current.push({
          x: currentX,
          y: barY,
          width: barWidth - 2,
          height: barHeight,
          event: event,
        });

        // Bar color
        const isPositive = event.impact >= 0;
        const baseColor = isPositive
          ? 'rgba(76, 175, 80, 0.9)'
          : 'rgba(244, 67, 54, 0.9)';

        ctx.fillStyle = baseColor;
        ctx.fillRect(currentX, barY, barWidth - 2, Math.abs(barHeight));

        // Side border
        ctx.strokeStyle = side === 'ct' ? '#2196f3' : '#ff9800';
        ctx.lineWidth = 2;
        ctx.strokeRect(currentX, barY, barWidth - 2, Math.abs(barHeight));

        // Impact value on bar
        if (Math.abs(barHeight) > 20) {
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 10px Inter, sans-serif';
          ctx.textAlign = 'center';
          const textY = isPositive ? barY + 12 : barY + Math.abs(barHeight) - 4;
          ctx.fillText(
            event.impact.toFixed(1),
            currentX + barWidth / 2 - 1,
            textY
          );
        }

        // Game state label below bar - only show if bar is wide enough
        if (barWidth > 20) {
          ctx.fillStyle = '#94a3b8';
          ctx.font = '8px Inter, sans-serif';
          ctx.textAlign = 'left';
          ctx.save();
          ctx.translate(
            currentX + barWidth / 2 - 1,
            padding.top + chartHeight + 8
          );
          ctx.rotate(-Math.PI / 3); // 60 degrees
          // Shorten long game state labels
          const stateLabel = event.game_state
            .replace(' saving', 's')
            .replace('v', ':');
          ctx.fillText(stateLabel, 0, 0);
          ctx.restore();
        }

        currentX += barWidth + 2;
      });

      // Round separator
      if (roundIndex < rounds.length - 1) {
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(currentX + roundGap / 2, padding.top);
        ctx.lineTo(currentX + roundGap / 2, padding.top + chartHeight);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Round label at top
      const roundCenterX = roundStartX + roundWidth / 2;
      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Round ${roundNum}`, roundCenterX, padding.top - 35);

      // Round delta (total impact for round)
      const roundImpact = events.reduce((sum, e) => sum + e.impact, 0);
      ctx.fillStyle = roundImpact >= 0 ? '#4caf50' : '#f44336';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.fillText(
        `Δ ${roundImpact >= 0 ? '+' : ''}${roundImpact.toFixed(1)}`,
        roundCenterX,
        padding.top - 18
      );

      // Events count
      ctx.fillStyle = '#94a3b8';
      ctx.font = '9px Inter, sans-serif';
      ctx.fillText(
        `${events.length} event${events.length > 1 ? 's' : ''}`,
        roundCenterX,
        padding.top - 5
      );

      currentX += roundGap;
    });

    // Title

    // Y-axis label
    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Impact Score', 0, 0);
    ctx.restore();

    // X-axis label
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Individual Events by Round', width / 2, height - 10);

    // Legend
    const legendX = width - padding.right - 200;
    const legendY = 30;

    // Positive impact
    ctx.fillStyle = 'rgba(76, 175, 80, 0.9)';
    ctx.fillRect(legendX, legendY, 15, 15);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Positive Impact', legendX + 20, legendY + 11);

    // Negative impact
    ctx.fillStyle = 'rgba(244, 67, 54, 0.9)';
    ctx.fillRect(legendX + 120, legendY, 15, 15);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText('Negative Impact', legendX + 140, legendY + 11);

    // CT/T side indicators
    ctx.strokeStyle = '#2196f3';
    ctx.lineWidth = 3;
    ctx.strokeRect(legendX, legendY + 25, 15, 15);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText('CT Side', legendX + 20, legendY + 36);

    ctx.strokeStyle = '#ff9800';
    ctx.strokeRect(legendX + 120, legendY + 25, 15, 15);
    ctx.fillText('T Side', legendX + 140, legendY + 36);
  }, [data, playerName, width, height]);

  return (
    <div style={{ position: 'relative' }}>
      <div
        style={{
          overflowX: 'auto',
          background: '#0f172a',
          borderRadius: '8px',
          padding: '10px',
        }}
      >
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{
            display: 'block',
            maxWidth: '100%',
            cursor: 'crosshair',
          }}
        />
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          style={{
            position: 'fixed',
            left: tooltip.x + 10,
            top: tooltip.y + 10,
            background: 'rgba(15, 23, 42, 0.98)',
            border: '2px solid #334155',
            borderRadius: '8px',
            padding: '12px',
            color: '#e2e8f0',
            fontSize: '12px',
            pointerEvents: 'none',
            zIndex: 1000,
            minWidth: '220px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
          }}
        >
          <div
            style={{
              fontWeight: 'bold',
              marginBottom: '8px',
              fontSize: '14px',
              color: '#fff',
            }}
          >
            Round {tooltip.event.round} - {tooltip.event.side.toUpperCase()}
          </div>
          <div style={{ display: 'grid', gap: '4px' }}>
            <div>
              <span style={{ color: '#94a3b8' }}>Game State:</span>{' '}
              <span style={{ color: '#e2e8f0', fontWeight: '500' }}>
                {tooltip.event.game_state}
              </span>
            </div>
            <div>
              <span style={{ color: '#94a3b8' }}>Impact:</span>{' '}
              <span
                style={{
                  color: tooltip.event.impact >= 0 ? '#4caf50' : '#f44336',
                  fontWeight: 'bold',
                }}
              >
                {tooltip.event.impact >= 0 ? '+' : ''}
                {tooltip.event.impact.toFixed(1)}
              </span>
            </div>
            <div>
              <span style={{ color: '#94a3b8' }}>Pre Win %:</span>{' '}
              <span style={{ color: '#e2e8f0' }}>
                {(tooltip.event.pre_win * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span style={{ color: '#94a3b8' }}>Post Win %:</span>{' '}
              <span style={{ color: '#e2e8f0' }}>
                {(tooltip.event.post_win * 100).toFixed(1)}%
              </span>
            </div>
            {tooltip.event.post_plant && (
              <div
                style={{
                  color: '#ff9800',
                  fontWeight: '500',
                  marginTop: '4px',
                }}
              >
                💣 Post-plant situation
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default ImpactCanvasChart;
