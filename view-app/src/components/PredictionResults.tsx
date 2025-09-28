import type { PredictionWithBinning } from '../types';
import './PredictionResults.css';

interface PredictionResultsProps {
  predictions: PredictionWithBinning[];
}

export default function PredictionResults({
  predictions,
}: PredictionResultsProps) {
  if (predictions.length === 0) {
    return null;
  }

  // Unified display for all predictions
  const binningData = predictions[0]?.binningData;
  const sampleCount = binningData?.n_samples || 0;
  const binningCtProb =
    binningData && sampleCount > 0
      ? (binningData.ct_win_probability * 100).toFixed(1)
      : 0;

  return (
    <div className='comparison-container'>
      <div className='comparison-bars-container'>
        {binningData && (
          <div className='comparison-model-bar'>
            <div className='model-label'>
              Training Data
              {sampleCount < 50 && (
                <span className='low-sample-warning'> ⚠️</span>
              )}
            </div>
            <div
              className='ct-bar-container'
              style={{ backgroundColor: sampleCount == 0 ? 'red' : '' }}
            >
              <div
                className='ct-bar-fill'
                style={{ width: `${binningCtProb}%` }}
              >
                <span className='ct-percentage'>{binningCtProb}%</span>
              </div>
            </div>
            <div className='model-stats-compact'>{sampleCount} samples</div>
          </div>
        )}

        {predictions.map((result) => {
          const ctProb = (result.prediction.ct_win_probability * 100).toFixed(
            1
          );
          const modelDisplayName = result.modelName
            .replace('xgboost_', '')
            .replace('_', ' ');

          return (
            <div key={result.modelName} className='comparison-model-bar'>
              <div className='model-label'>
                {modelDisplayName}
                {` ${ctProb}%`}
              </div>
              <div className='ct-bar-container'>
                <div className='ct-bar-fill' style={{ width: `${ctProb}%` }}>
                  <span className='ct-percentage'>{ctProb}%</span>
                </div>
              </div>
              <div className='model-stats-compact'>
                {(result.model_info.accuracy * 100).toFixed(1)}% acc
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
