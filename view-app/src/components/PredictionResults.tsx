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
  const binningCtProb = binningData
    ? (binningData.ct_win_probability * 100).toFixed(1)
    : null;
  const binningTProb = binningData
    ? (binningData.t_win_probability * 100).toFixed(1)
    : null;
  const sampleCount = binningData?.n_samples || 0;

  return (
    <div className='comparison-container'>
      <div className='comparison-header'>
        <h3>Prediction Results</h3>
        <p className='comparison-subtitle'>
          {predictions.length} model{predictions.length > 1 ? 's' : ''}{' '}
          {binningData ? '+ training data' : ''}
        </p>
      </div>

      <div className='comparison-grid'>
        {binningData && (
          <div className='model-prediction-card binning-card'>
            <div className='model-prediction-header'>
              <h4>📊 Training Data</h4>
              <div className='model-mini-stats'>
                <span>Samples: {sampleCount}</span>
                {sampleCount < 50 && (
                  <span className='low-sample-warning'>⚠️ Low</span>
                )}
              </div>
            </div>

            <div className='prediction-winner'>
              <strong>{parseFloat(binningCtProb!) > 50 ? 'CT' : 'T'}</strong>
              <span className='confidence-mini'>Historical</span>
            </div>

            <div className='probability-bars-mini'>
              <div className='prob-bar-mini'>
                <label>CT</label>
                <div className='progress-bar-mini'>
                  <div
                    className='progress ct-progress'
                    style={{ width: `${binningCtProb}%` }}
                  ></div>
                  <span className='percentage-mini'>{binningCtProb}%</span>
                </div>
              </div>

              <div className='prob-bar-mini'>
                <label>T</label>
                <div className='progress-bar-mini'>
                  <div
                    className='progress t-progress'
                    style={{ width: `${binningTProb}%` }}
                  ></div>
                  <span className='percentage-mini'>{binningTProb}%</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {predictions.map((result) => {
          const ctProb = (result.prediction.ct_win_probability * 100).toFixed(
            1
          );
          const tProb = (result.prediction.t_win_probability * 100).toFixed(1);
          const confidence = (result.prediction.confidence * 100).toFixed(1);
          const modelDisplayName = result.modelName
            .replace('xgboost_', '')
            .replace('_', ' ');

          // Calculate difference from binning data
          const ctDiff = binningData
            ? (result.prediction.ct_win_probability -
                binningData.ct_win_probability) *
              100
            : 0;

          return (
            <div key={result.modelName} className='model-prediction-card'>
              <div className='model-prediction-header'>
                <h4>{modelDisplayName}</h4>
                <div className='model-mini-stats'>
                  <span>
                    Acc: {(result.model_info.accuracy * 100).toFixed(1)}%
                  </span>
                  <span>AUC: {result.model_info.auc.toFixed(3)}</span>
                </div>
              </div>

              <div className='prediction-winner'>
                <strong>{result.prediction.predicted_winner}</strong>
                <span className='confidence-mini'>Conf: {confidence}%</span>
                {binningData && (
                  <span
                    className={`diff-indicator ${
                      ctDiff > 0 ? 'positive' : 'negative'
                    }`}
                  >
                    {ctDiff > 0 ? '+' : ''}
                    {ctDiff.toFixed(1)}pp
                  </span>
                )}
              </div>

              <div className='probability-bars-mini'>
                <div className='prob-bar-mini'>
                  <label>CT</label>
                  <div className='progress-bar-mini'>
                    <div
                      className='progress ct-progress'
                      style={{ width: `${ctProb}%` }}
                    ></div>
                    <span className='percentage-mini'>{ctProb}%</span>
                  </div>
                </div>

                <div className='prob-bar-mini'>
                  <label>T</label>
                  <div className='progress-bar-mini'>
                    <div
                      className='progress t-progress'
                      style={{ width: `${tProb}%` }}
                    ></div>
                    <span className='percentage-mini'>{tProb}%</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
