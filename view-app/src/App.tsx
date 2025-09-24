import { useState, useEffect, useRef } from 'react';
import './App.css';
import ModelSelection from './components/ModelSelection';
import FeatureInputs from './components/FeatureInputs';
import PredictionResults from './components/PredictionResults';
import ScenarioButtons from './components/ScenarioButtons';
import { API_ENDPOINTS } from './config/api';
import { updateCalculatedStats } from './utils/playerStatsCalculator';
import type { Model, Feature, PredictionResult, FeatureValues } from './types';

function App() {
  const [models, setModels] = useState<{ [key: string]: Model }>({});
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [features, setFeatures] = useState<Feature[]>([]);
  const [featureValues, setFeatureValues] = useState<FeatureValues>({});
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const timeout = useRef<any>(null);

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
        allFeatures.forEach((feature) => {
          initialValues[feature.name] =
            typeof feature.default === 'boolean'
              ? feature.default
                ? 1
                : 0
              : Number(feature.default);
        });

        // Calculate team stats from initial player data
        const calculatedInitialValues = updateCalculatedStats(initialValues);
        setFeatureValues(calculatedInitialValues);
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
      ...featureValues,
      [featureName]: value,
    };

    const updatedVals = updateCalculatedStats(newVals);
    setFeatureValues(updatedVals);

    if (timeout.current) clearTimeout(timeout.current);

    timeout.current = setTimeout(() => {
      handlePredict(updatedVals);
    }, 200);
  };

  const handlePredict = async (inputFeatureVals?: FeatureValues) => {
    if (selectedModels.length === 0) return;

    setError(null);

    try {
      const predictionPromises = selectedModels.map(async (modelName) => {
        const response = await fetch(API_ENDPOINTS.predict, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model_name: modelName,
            features: inputFeatureVals ?? featureValues,
          }),
        });

        const data = await response.json();
        if (!data.success) throw new Error(data.error);
        return { modelName, ...data };
      });

      const results = await Promise.all(predictionPromises);
      setPredictions(results);
    } catch (err) {
      setError('Prediction failed: ' + (err as Error).message);
    } finally {
    }
  };

  const handleScenarioLoad = (scenarioData: { [key: string]: number }) => {
    setFeatureValues((prev) => {
      const updated = {
        ...prev,
        ...scenarioData,
      };

      // Automatically calculate team stats from player data
      return updateCalculatedStats(updated);
    });
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
                selectedModels={selectedModels}
                onFeatureValueChange={handleFeatureValueChange}
                isLoading={false}
              />

              <ScenarioButtons
                onScenarioLoad={handleScenarioLoad}
                disabled={selectedModels.length === 0}
              />
            </>
          )}
        </div>

        {selectedModels.length > 0 && (
          <section className='prediction-section'>
            <h2>🎯 Prediction</h2>
            <button
              className={`predict-button`}
              onClick={() => handlePredict()}
              disabled={selectedModels.length === 0}
            >
              {selectedModels.length > 1
                ? 'Compare Model Predictions'
                : 'Predict Win Probability'}
            </button>

            {error && <div className='error'>{error}</div>}

            <PredictionResults predictions={predictions} />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
