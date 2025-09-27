import { useState, useEffect, useRef } from 'react';
import './App.css';
import ModelSelection from './components/ModelSelection';
import FeatureInputs from './components/FeatureInputs';
import PredictionResults from './components/PredictionResults';
import { API_ENDPOINTS } from './config/api';
import { updateCalculatedStats } from './utils/playerStatsCalculator';
import type {
  Model,
  Feature,
  FeatureValues,
  BinningValues,
  PredictionWithBinning,
} from './types';

function App() {
  const [models, setModels] = useState<{ [key: string]: Model }>({});
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [features, setFeatures] = useState<Feature[]>([]);
  const [featureValuesRaw, setFeatureValues] = useState<FeatureValues>({});
  const [binningValues, setBinningValues] = useState<BinningValues>({});
  const [predictions, setPredictions] = useState<PredictionWithBinning[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [shouldPredict, setShouldPredict] = useState<number>(0);
  const timeout = useRef<any>(null);

  const featureValues = updateCalculatedStats(featureValuesRaw);

  // Load models on component mount
  useEffect(() => {
    fetch(API_ENDPOINTS.models)
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          const inputModels: { [modelName: string]: Model } = {};
          Object.values(data.models as { [key: string]: Model }).forEach(
            (m) => (inputModels[m.name] = m)
          );
          setModels(inputModels);
        } else {
          setError('Failed to load models');
        }
      })
      .catch((err) => {
        setError('Error loading models: ' + err.message);
      });
  }, []);

  const predictDebounce = (featureVals: FeatureValues) => {
    if (timeout.current) clearTimeout(timeout.current);
    timeout.current = setTimeout(() => {
      setShouldPredict((v) => v + 1);
    }, 200);
  };

  // Load features when models are selected
  useEffect(() => {
    if (selectedModels.length === 0) {
      setFeatures([]);
      setFeatureValues({});
      setPredictions([]);
      return;
    }

    const featurePromises = selectedModels.map(async (modelName) => {
      const response = await fetch(API_ENDPOINTS.modelFeatures(modelName));
      const data = await response.json();
      if (!data.success) throw new Error(data.error);
      return { modelName, features: data.features };
    });

    Promise.all(featurePromises)
      .then((allModelFeatures) => {
        // Combine all features from all models
        const featureMap = new Map();
        allModelFeatures.forEach((modelData) => {
          modelData.features.forEach((feature: Feature) => {
            if (!featureMap.has(feature.name)) {
              featureMap.set(feature.name, feature);
            }
          });
        });

        const allFeatures = Array.from(featureMap.values());
        setFeatures(allFeatures);

        // Initialize feature values with defaults
        const initialValues: FeatureValues = {};
        const initialBinning: BinningValues = {};
        allFeatures.forEach((feature) => {
          initialValues[feature.name] =
            typeof feature.default === 'boolean'
              ? feature.default
                ? 1
                : 0
              : Number(feature.default);

          // Initialize binning values with reasonable defaults
          if (feature.constraints.type === 'number') {
            const range =
              (feature.constraints.max || 100) - (feature.constraints.min || 0);
            initialBinning[feature.name] = Math.max(1, Math.round(range * 0.1)); // 10% of range
          } else {
            initialBinning[feature.name] = feature.default; // No binning for non-numeric features
          }
        });

        // Calculate team stats from initial player data
        setFeatureValues(initialValues);
        setBinningValues(initialBinning);
        predictDebounce(updateCalculatedStats(initialValues));
      })
      .catch((err) => {
        setError('Error loading features: ' + err.message);
      })
      .finally(() => {});
  }, [selectedModels]);

  const handleModelSelectionChange = (modelName: string, selected: boolean) => {
    if (selected) {
      setSelectedModels((prev) => [...prev, modelName]);
    } else {
      setSelectedModels((prev) => prev.filter((m) => m !== modelName));
    }
  };

  const handleFeatureValueChange = (featureName: string, value: number) => {
    const newVals = {
      ...featureValuesRaw,
      [featureName]: value,
    };

    setFeatureValues(newVals);

    predictDebounce(
      updateCalculatedStats({
        ...featureValuesRaw,
        [featureName]: value,
      })
    );
  };

  const handleBinningValueChange = (featureName: string, value: number) => {
    setBinningValues((prev) => ({
      ...prev,
      [featureName]: value,
    }));

    // Trigger prediction with updated binning
    predictDebounce(featureValues);
  };

  useEffect(() => {
    handlePredict();
  }, [shouldPredict]);

  const handlePredict = async (inputFeatureVals?: FeatureValues) => {
    if (selectedModels.length === 0) return;
    console.log(binningValues['bomb_planted']);

    setError(null);

    try {
      // Prepare features for binning comparison
      const currentFeatures = inputFeatureVals ?? featureValues;

      // Create features with bins for slice_dataset call
      const featuresWithBins: {
        [key: string]: { value: number; bin_size: number };
      } = {};
      Object.keys(currentFeatures).forEach((featureName) => {
        if (isNaN(currentFeatures[featureName])) return;
        featuresWithBins[featureName] = {
          value: currentFeatures[featureName],
          bin_size: binningValues[featureName] ?? 0,
        };
      });

      // Make both prediction and slice_dataset calls
      const [predictionResults, sliceDatasetResponse] = await Promise.all([
        // ML Model predictions
        Promise.all(
          selectedModels.map(async (modelName) => {
            const response = await fetch(API_ENDPOINTS.predict, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                model_name: modelName,
                features: currentFeatures,
              }),
            });

            const data = await response.json();
            if (!data.success) throw new Error(data.error);
            return { modelName, ...data };
          })
        ),

        // Raw dataset slice
        fetch(API_ENDPOINTS.sliceDataset, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            features_with_bins: featuresWithBins,
          }),
        }).then((response) => response.json()),
      ]);

      // Combine predictions with binning data
      const combinedResults: PredictionWithBinning[] = predictionResults.map(
        (result) => ({
          ...result,
          binningData: sliceDatasetResponse,
        })
      );

      setPredictions(combinedResults);
    } catch (err) {
      setError('Prediction failed: ' + (err as Error).message);
    } finally {
    }
  };

  return (
    <div className='app'>
      <header className='header'>
        <h1>🎮 CS2 Win Probability Predictor</h1>
        <p>
          Predict CT team win probability using advanced machine learning models
        </p>
      </header>

      <main className='main-content'>
        <div className='scrollable-content'>
          <ModelSelection
            models={models}
            selectedModels={selectedModels}
            onSelectionChange={handleModelSelectionChange}
          />

          {selectedModels.length > 0 && (
            <>
              <FeatureInputs
                features={features}
                featureValues={featureValues}
                binningValues={binningValues}
                selectedModels={selectedModels}
                onFeatureValueChange={handleFeatureValueChange}
                onBinningValueChange={handleBinningValueChange}
                isLoading={false}
              />
            </>
          )}
        </div>

        {selectedModels.length > 0 && (
          <section className='prediction-section'>
            <h2>🎯 Prediction</h2>

            {error && <div className='error'>{error}</div>}

            <PredictionResults predictions={predictions} />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
