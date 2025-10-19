import type { Feature, FeatureValues, BinningValues } from '../types';
import { isCalculatedTeamStat } from '../utils/playerStatsCalculator';
import './FeatureInputs.css';

interface FeatureInputsProps {
  features: Feature[];
  featureValues: FeatureValues;
  binningValues: BinningValues;
  selectedModels: string[];
  onFeatureValueChange: (featureName: string, value: number | string) => void;
  onBinningValueChange: (featureName: string, value: number) => void;
  isLoading: boolean;
}

export default function FeatureInputs({
  features,
  featureValues,
  binningValues,
  selectedModels,
  onFeatureValueChange,
  onBinningValueChange,
  isLoading,
}: FeatureInputsProps) {
  const handleRangeChange = (featureName: string, value: string) => {
    onFeatureValueChange(featureName, parseFloat(value));
  };

  const handleCheckboxChange = (featureName: string, checked: boolean) => {
    onFeatureValueChange(featureName, checked ? 1 : 0);
  };

  const handleBinningChange = (featureName: string, value: string) => {
    onBinningValueChange(featureName, parseFloat(value));
  };

  if (isLoading) {
    return (
      <section className='feature-section'>
        <h2>Game State Features</h2>
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
    const isCalculated =
      isCalculatedTeamStat(feature.name) && hasPlayerFeatures;

    if (feature.constraints.type === 'checkbox') {
      const currentBinValue = binningValues[feature.name] ?? 0;

      return (
        <div
          key={feature.name}
          className={`feature-input checkbox-input ${
            isCalculated ? 'calculated' : ''
          }`}
        >
          <div className='feature-main-control'>
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
        </div>
      );
    }

    if (feature.constraints.type === 'select') {
      const currentBinValue = binningValues[feature.name] ?? 0;

      return (
        <div
          key={feature.name}
          className={`feature-input ${isCalculated ? 'calculated' : ''}`}
        >
          <div className='feature-main-control'>
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
                onFeatureValueChange(feature.name, e.target.value)
              }
              disabled={isCalculated}
              title={
                isCalculated
                  ? 'This value is automatically calculated from player data'
                  : ''
              }
            >
              {feature.constraints.options?.map((value) => {
                if (typeof value === 'string') {
                  return (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  );
                }
              })}
            </select>
          </div>
          <div className='binning-control'>
            <span className='binning-info'>
              {currentBinValue === 0
                ? 'Exact match only'
                : 'No binning for select'}
            </span>
          </div>
        </div>
      );
    }

    // Range input
    const currentBinValue = binningValues[feature.name] ?? 0;
    const maxBinSize =
      ((feature.constraints.max || 100) - (feature.constraints.min || 0)) / 2;

    return (
      <div
        key={feature.name}
        className={`feature-input feature-input-with-binning ${
          isCalculated ? 'calculated' : ''
        }`}
      >
        <div className='feature-main-control'>
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
        <div className='binning-control'>
          <label htmlFor={`${feature.name}_bin`} className='binning-label'>
            Bin size: <span className='bin-display'>{currentBinValue}</span>
          </label>
          <input
            type='range'
            id={`${feature.name}_bin`}
            min={-1}
            max={maxBinSize}
            step={feature.constraints.step || 1}
            value={currentBinValue}
            onChange={(e) => handleBinningChange(feature.name, e.target.value)}
            className='binning-slider'
            title='Range around the value to include in dataset comparison (±bin size)'
          />
        </div>
      </div>
    );
  };

  return (
    <section className='feature-section'>
      {Object.entries(groups).map(([groupName, groupFeatures]) => {
        if (groupFeatures.length === 0) return null;

        return (
          <div key={groupName} className='feature-group'>
            <h3 className='group-title'>{groupName}</h3>
            <div className='feature-grid'>
              {groupFeatures
                .sort((a, b) => {
                  // CTs first
                  const aIsCt = a.display_name.includes('Ct');
                  const bIsCt = b.display_name.includes('Ct');
                  if (aIsCt !== bIsCt) {
                    return aIsCt ? -1 : 1;
                  }

                  // Then Ts
                  const aIsT =
                    a.display_name.includes('Ts ') ||
                    a.display_name.includes('T ') ||
                    (a.display_name.includes(' T') &&
                      !a.display_name.includes('Time'));
                  const bIsT =
                    b.display_name.includes('Ts ') ||
                    b.display_name.includes('T ') ||
                    (b.display_name.includes(' T') &&
                      !b.display_name.includes('Time'));
                  if (aIsT !== bIsT) {
                    return aIsT ? -1 : 1;
                  }

                  const aIsTime = a.display_name.includes('Round Time');
                  const bIsTime = b.display_name.includes('Round Time');
                  if (aIsTime !== bIsTime) {
                    return aIsTime ? -1 : 1;
                  }

                  // Then Bombs
                  const aIsBomb = a.display_name.includes('Bomb');
                  const bIsBomb = b.display_name.includes('Bomb');
                  if (aIsBomb !== bIsBomb) {
                    return aIsBomb ? -1 : 1;
                  }

                  return 0;
                })
                .map(renderFeatureInput)}
            </div>
          </div>
        );
      })}
    </section>
  );
}
