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

  if (predictions.length === 1) {
    const result = predictions[0];
    const ctProb = (result.prediction.ct_win_probability * 100).toFixed(1);
    const tProb = (result.prediction.t_win_probability * 100).toFixed(1);
    const confidence = (result.prediction.confidence * 100).toFixed(1);

    // Binning data from raw dataset
    const binningCtProb = result.binningData
      ? (result.binningData.ct_win_probability * 100).toFixed(1)
      : null;
    const binningTProb = result.binningData
      ? (result.binningData.t_win_probability * 100).toFixed(1)
      : null;
    const sampleCount = result.binningData?.n_samples || 0;

    return (
      <div className='prediction-card'>
        <div className='prediction-header'>
          <h3>Prediction Results</h3>
          <span className='model-name'>
            {result.model_info.name.replace('xgboost_', '').replace('_', ' ')}
          </span>
        </div>

        <div className='prediction-comparison'>
          <div className='prediction-column'>
            <h4>🤖 ML Model Prediction</h4>
            <div className='probability-bars'>
              <div className='prob-bar'>
                <label>CT Win Probability</label>
                <div className='progress-bar'>
                  <div
                    className='progress ct-progress'
                    style={{ width: `${ctProb}%` }}
                  ></div>
                  <span className='percentage'>{ctProb}%</span>
                </div>
              </div>

              <div className='prob-bar'>
                <label>T Win Probability</label>
                <div className='progress-bar'>
                  <div
                    className='progress t-progress'
                    style={{ width: `${tProb}%` }}
                  ></div>
                  <span className='percentage'>{tProb}%</span>
                </div>
              </div>
            </div>

            <div className='prediction-summary'>
              <div className='winner'>
                <strong>
                  Predicted Winner: {result.prediction.predicted_winner}
                </strong>
              </div>
              <div className='confidence'>Confidence: {confidence}%</div>
            </div>

            <div className='model-stats'>
              <small>
                Model Accuracy: {(result.model_info.accuracy * 100).toFixed(1)}%
                | AUC: {result.model_info.auc.toFixed(3)} | Features:{' '}
                {result.model_info.feature_count}
              </small>
            </div>
          </div>

          {result.binningData && (
            <div className='prediction-column'>
              <h4>📊 Training Data (Binned)</h4>
              <div className='probability-bars'>
                <div className='prob-bar'>
                  <label>CT Win Probability</label>
                  <div className='progress-bar'>
                    <div
                      className='progress ct-progress'
                      style={{ width: `${binningCtProb}%` }}
                    ></div>
                    <span className='percentage'>{binningCtProb}%</span>
                  </div>
                </div>

                <div className='prob-bar'>
                  <label>T Win Probability</label>
                  <div className='progress-bar'>
                    <div
                      className='progress t-progress'
                      style={{ width: `${binningTProb}%` }}
                    ></div>
                    <span className='percentage'>{binningTProb}%</span>
                  </div>
                </div>
              </div>

              <div className='prediction-summary'>
                <div className='winner'>
                  <strong>
                    Historical Winner:{' '}
                    {parseFloat(binningCtProb!) > 50 ? 'CT' : 'T'}
                  </strong>
                </div>
                <div className='sample-count'>
                  Sample Size: {sampleCount} rounds
                </div>
              </div>

              <div className='binning-info'>
                <small>
                  Based on similar situations in training data
                  {sampleCount < 50 && (
                    <span className='low-sample-warning'>
                      {' '}
                      ⚠️ Low sample size
                    </span>
                  )}
                </small>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Multiple models comparison
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
        <h3>Model Comparison Results</h3>
        <p className='comparison-subtitle'>
          {predictions.length} models compared
        </p>
      </div>

      {binningData && (
        <div className='binning-summary'>
          <h4>📊 Training Data Reference</h4>
          <div className='binning-stats'>
            <span>CT: {binningCtProb}%</span>
            <span>T: {binningTProb}%</span>
            <span>Samples: {sampleCount}</span>
            {sampleCount < 50 && (
              <span className='low-sample-warning'>⚠️ Low sample size</span>
            )}
          </div>
        </div>
      )}

      <div className='comparison-grid'>
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
