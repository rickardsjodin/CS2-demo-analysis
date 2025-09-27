import { useState, useEffect, useRef } from 'react';
import './App.css';
import ModelSelection from './components/ModelSelection';
import FeatureInputs from './components/FeatureInputs';
import PredictionResults from './components/PredictionResults';
import FeatureRangeAnalyzer from './components/FeatureRangeAnalyzer';
import Plot2D from './components/Plot2D';
import { API_ENDPOINTS } from './config/api';
import { applyContraints } from './utils/playerStatsCalculator';
import type {
  Model,
  Feature,
  FeatureValues,
  BinningValues,
  PredictionWithBinning,
  FeatureRangeAnalysis,
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
  const [rangeAnalysis, setRangeAnalysis] =
    useState<FeatureRangeAnalysis | null>(null);
  const timeout = useRef<any>(null);

  const featureValues = applyContraints(featureValuesRaw);

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

  const predictDebounce = () => {
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
            initialBinning[feature.name] = -1; // Math.max(1, Math.round(range * 0.1)); // 10% of range
          } else {
            initialBinning[feature.name] = feature.default; // No binning for non-numeric features
          }
        });

        // Calculate team stats from initial player data
        setFeatureValues(initialValues);
        setBinningValues(initialBinning);
        predictDebounce();
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

    predictDebounce();
  };

  const handleBinningValueChange = (featureName: string, value: number) => {
    setBinningValues((prev) => ({
      ...prev,
      [featureName]: value,
    }));

    // Trigger prediction with updated binning
    predictDebounce();
  };

  useEffect(() => {
    handlePredict();
  }, [shouldPredict]);

  const handlePredict = async (inputFeatureVals?: FeatureValues) => {
    if (selectedModels.length === 0) return;

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

  const handleRangeAnalysis = async (
    featureName: string
  ): Promise<FeatureRangeAnalysis> => {
    if (selectedModels.length === 0) {
      throw new Error('No models selected');
    }

    // Find the selected feature to get its constraints
    const selectedFeature = features.find((f) => f.name === featureName);
    if (!selectedFeature || selectedFeature.constraints.type !== 'number') {
      throw new Error('Only numeric features can be analyzed');
    }

    // Determine range and step size
    const minVal = selectedFeature.constraints.min ?? 0;
    const maxVal = selectedFeature.constraints.max ?? 100;
    const step = selectedFeature.constraints.step ?? 1;
    const featureValueRange: number[] = [];

    for (let i = minVal; i <= maxVal; i += step) {
      featureValueRange.push(i);
    }

    const analysis: FeatureRangeAnalysis = {
      featureName: featureName,
      models: [],
      datasetReference: undefined,
    };

    // Analyze each model
    for (const modelName of selectedModels) {
      const modelResults = {
        modelName: modelName,
        model_info: {
          name: modelName,
          accuracy: 0,
          auc: 0,
          feature_count: 0,
        },
        data: [] as Array<{
          featureValue: number;
          ct_win_probability: number;
          t_win_probability: number;
        }>,
      };

      // Make predictions for each feature value
      for (const featureVal of featureValueRange) {
        try {
          // Create feature set with the varying feature
          const testFeatures = {
            ...featureValues,
            [featureName]: featureVal,
          };

          const response = await fetch(API_ENDPOINTS.predict, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              model_name: modelName,
              features: applyContraints(testFeatures),
            }),
          });

          const data = await response.json();
          if (!data.success) {
            console.warn(
              `Prediction failed for ${modelName} at ${featureName}=${featureVal}: ${data.error}`
            );
            continue;
          }

          // Store model info from first successful prediction
          if (modelResults.model_info.accuracy === 0) {
            modelResults.model_info = data.model_info;
          }

          modelResults.data.push({
            featureValue: featureVal,
            ct_win_probability: data.prediction.ct_win_probability,
            t_win_probability: data.prediction.t_win_probability,
          });
        } catch (err) {
          console.warn(
            `Error predicting for ${modelName} at ${featureName}=${featureVal}:`,
            err
          );
        }
      }

      if (modelResults.data.length > 0) {
        analysis.models.push(modelResults);
      }
    }

    // Generate dataset reference using slice_dataset calls
    try {
      const datasetResults = {
        data: [] as Array<{
          featureValue: number;
          ct_win_probability: number;
          t_win_probability: number;
        }>,
        totalSamples: 0,
      };

      // Create features with bins for slice_dataset call
      const featuresWithBins: {
        [key: string]: { value: number; bin_size: number };
      } = {};
      Object.keys(featureValues).forEach((featureName) => {
        if (isNaN(featureValues[featureName])) return;
        featuresWithBins[featureName] = {
          value: featureValues[featureName],
          bin_size: binningValues[featureName] ?? 0,
        };
      });

      const binSize = binningValues[featureName] ?? 0;

      for (const featureVal of featureValueRange) {
        try {
          // Add the varying feature with its bin
          featuresWithBins[featureName] = {
            value: featureVal,
            bin_size: binSize,
          };

          const response = await fetch(API_ENDPOINTS.sliceDataset, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              features_with_bins: featuresWithBins,
            }),
          });

          const data = await response.json();
          if (data.n_samples > 0) {
            datasetResults.data.push({
              featureValue: featureVal,
              ct_win_probability: data.ct_win_probability,
              t_win_probability: data.t_win_probability,
            });
            datasetResults.totalSamples += data.n_samples;
          }
        } catch (err) {
          console.warn(
            `Error getting dataset reference for ${featureName}=${featureVal}:`,
            err
          );
        }
      }

      if (datasetResults.data.length > 0) {
        analysis.datasetReference = datasetResults;
      }
    } catch (err) {
      console.warn('Error generating dataset reference:', err);
    }

    setRangeAnalysis(analysis);
    return analysis;
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

              <div>
                <FeatureRangeAnalyzer
                  features={features}
                  featureValues={featureValues}
                  selectedModels={selectedModels}
                  onAnalyze={handleRangeAnalysis}
                />
                <Plot2D analysis={rangeAnalysis} />
              </div>
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
