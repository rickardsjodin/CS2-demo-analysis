import { useEffect, useRef } from 'react';
import type { FeatureRangeAnalysis } from '../types';
import './Plot2D.css';

interface Plot2DProps {
  analysis: FeatureRangeAnalysis | null;
  width?: number;
  height?: number;
}

interface PlotPoint {
  x: number;
  y: number;
  label: string;
  color: string;
}

export default function Plot2D({
  analysis,
  width = 800,
  height = 400,
}: Plot2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!analysis || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas dimensions
    canvas.width = width;
    canvas.height = height;

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Define margins and plot area
    const margin = { top: 40, right: 150, bottom: 60, left: 80 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Collect all data points
    const allData: PlotPoint[] = [];
    const colors = [
      '#2196F3',
      '#4CAF50',
      '#FF9800',
      '#9C27B0',
      '#F44336',
      '#795548',
      '#607D8B',
      '#E91E63',
      '#00BCD4',
      '#CDDC39',
    ];

    // Add model data
    analysis.models.forEach((model, index) => {
      const color = colors[index % colors.length];
      model.data.forEach((point) => {
        allData.push({
          x: point.featureValue,
          y: point.ct_win_probability * 100,
          label: model.modelName.replace('xgboost_', '').replace('_', ' '),
          color,
        });
      });
    });

    // Add dataset reference data
    if (analysis.datasetReference) {
      analysis.datasetReference.data.forEach((point) => {
        allData.push({
          x: point.featureValue,
          y: point.ct_win_probability * 100,
          label: 'Training Data',
          color: '#000000',
        });
      });
    }

    if (allData.length === 0) return;

    // Calculate scales
    const xValues = allData.map((d) => d.x);
    const yValues = allData.map((d) => d.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.max(0, Math.min(...yValues) - 5);
    const yMax = Math.min(100, Math.max(...yValues) + 5);

    const xScale = (x: number) =>
      margin.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
    const yScale = (y: number) =>
      margin.top + ((yMax - y) / (yMax - yMin)) * plotHeight;

    // Draw grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;

    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = margin.left + (i / 10) * plotWidth;
      ctx.beginPath();
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, margin.top + plotHeight);
      ctx.stroke();
    }

    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = margin.top + (i / 10) * plotHeight;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + plotWidth, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';

    // X-axis label
    ctx.fillText(
      analysis.featureName.replace('_', ' ').toUpperCase(),
      margin.left + plotWidth / 2,
      height - 20
    );

    // Y-axis label
    ctx.save();
    ctx.translate(20, margin.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('CT Win Probability (%)', 0, 0);
    ctx.restore();

    // Draw title
    ctx.font = 'bold 16px Arial';
    ctx.fillText(
      `Feature Range Analysis: ${analysis.featureName.replace('_', ' ')}`,
      width / 2,
      25
    );

    // Group data by label for line drawing
    const groupedData = new Map<string, PlotPoint[]>();
    allData.forEach((point) => {
      if (!groupedData.has(point.label)) {
        groupedData.set(point.label, []);
      }
      groupedData.get(point.label)!.push(point);
    });

    // Draw lines for each group
    groupedData.forEach((points) => {
      // Sort points by x value
      points.sort((a, b) => a.x - b.x);

      if (points.length < 2) return;

      ctx.strokeStyle = points[0].color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xScale(points[0].x), yScale(points[0].y));

      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(xScale(points[i].x), yScale(points[i].y));
      }
      ctx.stroke();
    });

    // Draw points
    allData.forEach((point) => {
      ctx.fillStyle = point.color;
      ctx.beginPath();
      ctx.arc(xScale(point.x), yScale(point.y), 4, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw X-axis ticks and labels
    ctx.fillStyle = '#333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
      const value = xMin + (i / 5) * (xMax - xMin);
      const x = margin.left + (i / 5) * plotWidth;

      // Tick mark
      ctx.beginPath();
      ctx.moveTo(x, margin.top + plotHeight);
      ctx.lineTo(x, margin.top + plotHeight + 5);
      ctx.stroke();

      // Label
      ctx.fillText(value.toFixed(1), x, margin.top + plotHeight + 20);
    }

    // Draw Y-axis ticks and labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const value = yMin + (i / 5) * (yMax - yMin);
      const y = margin.top + plotHeight - (i / 5) * plotHeight;

      // Tick mark
      ctx.beginPath();
      ctx.moveTo(margin.left - 5, y);
      ctx.lineTo(margin.left, y);
      ctx.stroke();

      // Label
      ctx.fillText(value.toFixed(1) + '%', margin.left - 10, y + 4);
    }

    // Draw legend
    ctx.textAlign = 'left';
    ctx.font = '12px Arial';
    const legendX = margin.left + plotWidth + 20;
    let legendY = margin.top + 20;

    const uniqueLabels = Array.from(new Set(allData.map((d) => d.label)));
    uniqueLabels.forEach((label) => {
      const color = allData.find((d) => d.label === label)?.color || '#000';

      // Legend color box
      ctx.fillStyle = color;
      ctx.fillRect(legendX, legendY - 8, 12, 12);

      // Legend text
      ctx.fillStyle = '#333';
      ctx.fillText(label, legendX + 20, legendY);

      legendY += 20;
    });
  }, [analysis, width, height]);

  if (!analysis) {
    return (
      <div className='plot2d-container'>
        <div className='plot2d-placeholder'>
          Select a feature and click "Analyze Range" to see the visualization
        </div>
      </div>
    );
  }

  return (
    <div className='plot2d-container'>
      <div className='plot2d-header'>
        <h3>📈 Feature Range Visualization</h3>
        <p>CT Win Probability vs {analysis.featureName.replace('_', ' ')}</p>
      </div>
      <div className='plot2d-canvas-container'>
        <canvas ref={canvasRef} className='plot2d-canvas' />
      </div>
    </div>
  );
}
