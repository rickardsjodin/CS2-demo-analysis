import type { Feature, FeatureValues } from '../types';
import { isCalculatedTeamStat } from '../utils/playerStatsCalculator';
import './FeatureInputs.css';

interface FeatureInputsProps {
  features: Feature[];
  featureValues: FeatureValues;
  selectedModels: string[];
  onFeatureValueChange: (featureName: string, value: number) => void;
  isLoading: boolean;
}

export default function FeatureInputs({
  features,
  featureValues,
  selectedModels,
  onFeatureValueChange,
  isLoading,
}: FeatureInputsProps) {
  const handleRangeChange = (featureName: string, value: string) => {
    onFeatureValueChange(featureName, parseFloat(value));
  };

  const handleCheckboxChange = (featureName: string, checked: boolean) => {
    onFeatureValueChange(featureName, checked ? 1 : 0);
  };

  if (isLoading) {
    return (
      <section className='feature-section'>
        <h2>⚙️ Game State Features</h2>
        <div className='loading'>Loading features...</div>
      </section>
    );
  }

  // Group features by category
  const groups = {
    'Game State': [] as Feature[],
    'Team Stats': [] as Feature[],
    Equipment: [] as Feature[],
    'Player Features': [] as Feature[],
  };

  let hasPlayerFeatures = false;
  let hasCalculatedFeatures = false;

  features.forEach((feature) => {
    if (feature.name.includes('player_')) {
      groups['Player Features'].push(feature);
      hasPlayerFeatures = true;
    } else if (
      feature.name.includes('time') ||
      feature.name.includes('bomb') ||
      feature.name.includes('alive')
    ) {
      groups['Game State'].push(feature);
    } else if (
      feature.name.includes('hp') ||
      feature.name.includes('armor') ||
      feature.name.includes('helmets')
    ) {
      groups['Team Stats'].push(feature);
      if (isCalculatedTeamStat(feature.name)) {
        hasCalculatedFeatures = true;
      }
    } else {
      groups['Equipment'].push(feature);
      if (isCalculatedTeamStat(feature.name)) {
        hasCalculatedFeatures = true;
      }
    }
  });

  const renderFeatureInput = (feature: Feature) => {
    const currentValue = featureValues[feature.name] ?? feature.default;
    const isCalculated = isCalculatedTeamStat(feature.name);

    if (feature.constraints.type === 'checkbox') {
      return (
        <div
          key={feature.name}
          className={`feature-input checkbox-input ${
            isCalculated ? 'calculated' : ''
          }`}
        >
          <input
            type='checkbox'
            id={feature.name}
            checked={currentValue === 1}
            onChange={(e) =>
              handleCheckboxChange(feature.name, e.target.checked)
            }
            disabled={isCalculated}
            title={
              isCalculated
                ? 'This value is automatically calculated from player data'
                : ''
            }
          />
          <label htmlFor={feature.name} className='checkbox-label'>
            {feature.display_name}
            {isCalculated && (
              <span className='calculated-indicator'> (auto)</span>
            )}
          </label>
        </div>
      );
    }

    if (feature.constraints.type === 'select') {
      return (
        <div
          key={feature.name}
          className={`feature-input ${isCalculated ? 'calculated' : ''}`}
        >
          <label htmlFor={feature.name}>
            {feature.display_name}:
            {isCalculated && (
              <span className='calculated-indicator'> (auto)</span>
            )}
          </label>
          <select
            id={feature.name}
            value={currentValue}
            onChange={(e) =>
              onFeatureValueChange(feature.name, parseFloat(e.target.value))
            }
            disabled={isCalculated}
            title={
              isCalculated
                ? 'This value is automatically calculated from player data'
                : ''
            }
          >
            {feature.constraints.options?.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>
      );
    }

    // Range input
    return (
      <div
        key={feature.name}
        className={`feature-input ${isCalculated ? 'calculated' : ''}`}
      >
        <label htmlFor={feature.name}>
          {feature.display_name}:{' '}
          <span className='value-display'>{currentValue}</span>
          {isCalculated && (
            <span className='calculated-indicator'> (auto)</span>
          )}
        </label>
        <input
          type='range'
          id={feature.name}
          min={feature.constraints.min}
          max={feature.constraints.max}
          step={feature.constraints.step}
          value={currentValue}
          onChange={(e) => handleRangeChange(feature.name, e.target.value)}
          disabled={isCalculated}
          title={
            isCalculated
              ? 'This value is automatically calculated from player data'
              : ''
          }
        />
      </div>
    );
  };

  return (
    <section className='feature-section'>
      <h2>⚙️ Game State Features</h2>

      <div className='selected-models-info'>
        <h3>
          {selectedModels.length > 1
            ? `Comparing ${selectedModels.length} models:`
            : `Model: ${selectedModels[0]
                ?.replace('xgboost_', '')
                .replace('_', ' ')}`}
        </h3>
        {selectedModels.length > 1 && (
          <div className='selected-models-list'>
            {selectedModels.map((name) => (
              <span key={name} className='model-tag'>
                {name.replace('xgboost_', '').replace('_', ' ')}
              </span>
            ))}
          </div>
        )}
        <p className='features-summary'>Showing {features.length} features</p>
        {hasPlayerFeatures && hasCalculatedFeatures && (
          <div className='auto-calculation-info'>
            <span className='info-icon'>ℹ️</span>
            Team stats are automatically calculated from individual player data
          </div>
        )}
      </div>

      {Object.entries(groups).map(([groupName, groupFeatures]) => {
        if (groupFeatures.length === 0) return null;

        return (
          <div key={groupName} className='feature-group'>
            <h3 className='group-title'>{groupName}</h3>
            <div className='feature-grid'>
              {groupFeatures.map(renderFeatureInput)}
            </div>
          </div>
        );
      })}
    </section>
  );
}
