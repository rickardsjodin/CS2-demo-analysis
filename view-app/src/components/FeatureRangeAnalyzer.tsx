import { useState } from 'react';
import type { Feature, FeatureValues, FeatureRangeAnalysis } from '../types';
import './FeatureRangeAnalyzer.css';

interface FeatureRangeAnalyzerProps {
  features: Feature[];
  featureValues: FeatureValues;
  selectedModels: string[];
  onAnalyze: (featureName: string) => Promise<FeatureRangeAnalysis>;
}

export default function FeatureRangeAnalyzer({
  features,
  featureValues,
  selectedModels,
  onAnalyze,
}: FeatureRangeAnalyzerProps) {
  const [selectedFeature, setSelectedFeature] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filter to only numeric features that can be ranged
  const numericFeatures = features.filter(
    (feature) => feature.constraints.type === 'number'
  );

  const handleAnalyze = async () => {
    if (!selectedFeature) {
      setError('Please select a feature to analyze');
      return;
    }

    if (selectedModels.length === 0) {
      setError('Please select at least one model');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      await onAnalyze(selectedFeature);
    } catch (err) {
      setError('Analysis failed: ' + (err as Error).message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (numericFeatures.length === 0) {
    return (
      <div className='feature-range-analyzer'>
        <div className='analyzer-message'>
          No numeric features available for range analysis
        </div>
      </div>
    );
  }

  return (
    <div className='feature-range-analyzer'>
      <div className='analyzer-header'>
        <h3>📊 Feature Range Analysis</h3>
        <p>Analyze how predictions change across the full range of a feature</p>
      </div>

      <div className='analyzer-controls'>
        <div className='feature-selection'>
          <label htmlFor='feature-select'>Select Feature:</label>
          <select
            id='feature-select'
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value)}
            disabled={isAnalyzing}
          >
            <option value=''>-- Choose a feature --</option>
            {numericFeatures.map((feature) => (
              <option key={feature.name} value={feature.name}>
                {feature.display_name}
              </option>
            ))}
          </select>
        </div>

        <button
          className='analyze-button'
          onClick={handleAnalyze}
          disabled={
            !selectedFeature || isAnalyzing || selectedModels.length === 0
          }
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze Range'}
        </button>
      </div>

      {selectedFeature && (
        <div className='feature-info'>
          <div className='current-value'>
            <span className='label'>Current Value:</span>
            <span className='value'>
              {featureValues[selectedFeature]?.toFixed(2) || 'N/A'}
            </span>
          </div>
          {(() => {
            const feature = numericFeatures.find(
              (f) => f.name === selectedFeature
            );
            return (
              feature && (
                <div className='feature-range'>
                  <span className='label'>Range:</span>
                  <span className='range'>
                    {feature.constraints.min ?? 0} -{' '}
                    {feature.constraints.max ?? 100}
                  </span>
                </div>
              )
            );
          })()}
        </div>
      )}

      {error && <div className='analyzer-error'>{error}</div>}
    </div>
  );
}
