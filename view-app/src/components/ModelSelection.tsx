import type { Model } from '../types';
import './ModelSelection.css';

interface ModelSelectionProps {
  models: { [key: string]: Model };
  selectedModels: string[];
  onSelectionChange: (modelName: string, selected: boolean) => void;
}

export default function ModelSelection({
  models,
  selectedModels,
  onSelectionChange,
}: ModelSelectionProps) {
  const handleCheckboxChange = (modelName: string, checked: boolean) => {
    onSelectionChange(modelName, checked);
  };

  return (
    <section className='model-selection'>
      <h2>📊 Select Models for Comparison</h2>
      <p className='selection-hint'>
        Select one or more models to compare their predictions
      </p>
      <div className='model-grid'>
        {Object.entries(models).map(([modelName, model]) => (
          <div
            key={modelName}
            className={`model-card ${
              selectedModels.includes(modelName) ? 'selected' : ''
            }`}
            onClick={() => {
              handleCheckboxChange(
                modelName,
                !selectedModels.includes(modelName)
              );
            }}
          >
            <div className='model-checkbox'>
              <input
                type='checkbox'
                id={`model_${modelName}`}
                className='model-selector'
                checked={selectedModels.includes(modelName)}
                onChange={(e) =>
                  handleCheckboxChange(modelName, e.target.checked)
                }
              />
            </div>
            <div className='model-header'>
              <h3>{model.display_name}</h3>
              <span className='model-badge'>
                {model.feature_count} features
              </span>
            </div>
            <div className='model-stats'>
              <div className='stat'>
                <label>Accuracy:</label>
                <span>{(model.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className='stat'>
                <label>AUC:</label>
                <span>{model.auc.toFixed(3)}</span>
              </div>
            </div>
            <div className='model-description'>{model.description}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
