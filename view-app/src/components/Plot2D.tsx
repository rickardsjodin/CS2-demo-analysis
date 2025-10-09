import { useEffect, useRef } from 'react';
import type {
  FeatureRangeAnalysis,
  PredictionWithBinning,
  FeatureValues,
} from '../types';
import './Plot2D.css';

interface Plot2DProps {
  analysis: FeatureRangeAnalysis | null;
  predictions?: PredictionWithBinning[];
  featureValues?: FeatureValues;
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
  predictions = [],
  featureValues = {},
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

    // Clear canvas with dark background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Define margins and plot area
    const margin = { top: 40, right: 150, bottom: 60, left: 80 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Collect all data points
    const allData: PlotPoint[] = [];
    // Brighter colors for dark theme
    const colors = [
      '#60a5fa', // blue
      '#4ade80', // green
      '#fb923c', // orange
      '#c084fc', // purple
      '#f87171', // red
      '#a78bfa', // violet
      '#22d3ee', // cyan
      '#f472b6', // pink
      '#fbbf24', // amber
      '#34d399', // emerald
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
          color: '#e2e8f0',
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

    // Check if this is a time-based feature that should be reversed (0 on the right)
    const shouldReverseXAxis =
      analysis.featureName === 'round_time_left' ||
      analysis.featureName === 'bomb_time_left';

    const xScale = (x: number) =>
      shouldReverseXAxis
        ? margin.left + plotWidth - ((x - xMin) / (xMax - xMin)) * plotWidth
        : margin.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
    const yScale = (y: number) =>
      margin.top + ((yMax - y) / (yMax - yMin)) * plotHeight;

    // Draw grid lines
    ctx.strokeStyle = '#334155';
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
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = '#e2e8f0';
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

    // Draw current predictions as live points (larger, with special styling)
    if (
      predictions.length > 0 &&
      featureValues[analysis.featureName] !== undefined
    ) {
      const currentFeatureValue = featureValues[analysis.featureName];

      predictions.forEach((prediction, index) => {
        const color = colors[index % colors.length];
        const x = xScale(currentFeatureValue);
        const y = yScale(prediction.prediction.ct_win_probability * 100);

        // Draw outer ring for emphasis
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.stroke();

        // Draw inner filled circle
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, 2 * Math.PI);
        ctx.fill();

        // Draw dark center dot
        ctx.fillStyle = '#0f172a';
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
      });

      // Draw vertical line to show current feature value
      ctx.strokeStyle = 'rgba(226, 232, 240, 0.3)';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(xScale(currentFeatureValue), margin.top);
      ctx.lineTo(xScale(currentFeatureValue), margin.top + plotHeight);
      ctx.stroke();
      ctx.setLineDash([]); // Reset dash pattern
    }

    // Draw X-axis ticks and labels
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
      // When reversed, labels should go from high to low
      const value = shouldReverseXAxis
        ? xMax - (i / 5) * (xMax - xMin)
        : xMin + (i / 5) * (xMax - xMin);
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
      const color = allData.find((d) => d.label === label)?.color || '#e2e8f0';

      // Legend color box
      ctx.fillStyle = color;
      ctx.fillRect(legendX, legendY - 8, 12, 12);

      // Legend text
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(label, legendX + 20, legendY);

      legendY += 20;
    });

    // Add current predictions to legend if they exist
    if (
      predictions.length > 0 &&
      featureValues[analysis.featureName] !== undefined
    ) {
      legendY += 10; // Add some spacing

      // Legend header for current predictions
      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 12px Arial';
      ctx.fillText('Current Predictions:', legendX, legendY);
      legendY += 20;

      ctx.font = '12px Arial';
      predictions.forEach((prediction, index) => {
        const color = colors[index % colors.length];
        const modelName = prediction.modelName
          .replace('xgboost_', '')
          .replace('_', ' ');

        // Legend symbol (circle with ring like the plot)
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(legendX + 6, legendY - 4, 6, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(legendX + 6, legendY - 4, 4, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#0f172a';
        ctx.beginPath();
        ctx.arc(legendX + 6, legendY - 4, 1, 0, 2 * Math.PI);
        ctx.fill();

        // Legend text with probability
        const probability = (
          prediction.prediction.ct_win_probability * 100
        ).toFixed(1);
        ctx.fillStyle = '#e2e8f0';
        ctx.fillText(`${modelName} (${probability}%)`, legendX + 20, legendY);

        legendY += 20;
      });
    }
  }, [analysis, predictions, featureValues, width, height]);

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
        <h3>Feature Range Visualization</h3>
        <p>CT Win Probability vs {analysis.featureName.replace('_', ' ')}</p>
      </div>
      <div className='plot2d-canvas-container'>
        <canvas ref={canvasRef} className='plot2d-canvas' />
      </div>
    </div>
  );
}
