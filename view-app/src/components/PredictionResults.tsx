import type { PredictionResult } from '../types';
import './PredictionResults.css';

interface PredictionResultsProps {
  predictions: PredictionResult[];
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

    return (
      <div className='prediction-card'>
        <div className='prediction-header'>
          <h3>Prediction Results</h3>
          <span className='model-name'>
            {result.model_info.name.replace('xgboost_', '').replace('_', ' ')}
          </span>
        </div>

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
            Model Accuracy: {(result.model_info.accuracy * 100).toFixed(1)}% |
            AUC: {result.model_info.auc.toFixed(3)} | Features:{' '}
            {result.model_info.feature_count}
          </small>
        </div>
      </div>
    );
  }

  // Multiple models comparison
  return (
    <div className='comparison-container'>
      <div className='comparison-header'>
        <h3>Model Comparison Results</h3>
        <p className='comparison-subtitle'>
          {predictions.length} models compared
        </p>
      </div>
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
