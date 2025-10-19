import { useState, useEffect, useRef, useMemo } from 'react';
import './App.css';
import ModelSelection from './components/ModelSelection';
import FeatureInputs from './components/FeatureInputs';
import PredictionResults from './components/PredictionResults';
import FeatureRangeAnalyzer from './components/FeatureRangeAnalyzer';
import Plot2D from './components/Plot2D';
import DemoAnalysis from './components/DemoAnalysis';
import { API_ENDPOINTS } from './config/api';
import { applyContraints } from './utils/playerStatsCalculator';
import type {
  Model,
  Feature,
  FeatureValues,
  BinningValues,
  PredictionWithBinning,
  FeatureRangeAnalysis,
  PlayerAnalysisEvent,
} from './types';

function App() {
  const [activeTab, setActiveTab] = useState<'prediction' | 'demo-analysis'>(
    'prediction'
  );

  // Prediction tab state
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

  // Demo analysis tab state
  const [demoId, setDemoId] = useState<string | null>(null);
  const [demoFilename, setDemoFilename] = useState<string | null>(null);
  const [demoPlayers, setDemoPlayers] = useState<string[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<
    PlayerAnalysisEvent[] | null
  >(null);

  const featureValues = useMemo(() => {
    const contrained = applyContraints(featureValuesRaw);
    const filtered: FeatureValues = {};
    features.forEach((f) => {
      filtered[f.name] = contrained[f.name];
    });
    return filtered;
  }, [featureValuesRaw, features]);

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

    const featureValsStr = localStorage.getItem('featureValuesRaw');
    if (featureValsStr) {
      const featureVals = JSON.parse(featureValsStr) as FeatureValues;
      setFeatureValues(featureVals);
    }
    const binningValsStr = localStorage.getItem('binningValues');
    if (binningValsStr) {
      setBinningValues(JSON.parse(binningValsStr) as BinningValues);
    }
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
            initialBinning[feature.name] = -1; // Math.max(1, Math.round(range * 0.1)); // 10% of range
          } else {
            initialBinning[feature.name] = 0; // feature.default; // No binning for non-numeric features
          }
        });

        // Calculate team stats from initial player data
        setFeatureValues((v) => ({
          ...initialValues,
          ...v,
        }));

        setBinningValues((v) => ({ ...initialBinning, ...v }));
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

  const handleFeatureValueChange = (
    featureName: string,
    value: number | string
  ) => {
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

    // Save in localstorage
    localStorage.setItem('featureValuesRaw', JSON.stringify(featureValues));
    localStorage.setItem('binningValues', JSON.stringify(binningValues));

    try {
      // Prepare features for binning comparison
      const currentFeatures = inputFeatureVals ?? featureValues;

      // Create features with bins for slice_dataset call
      const featuresWithBins: {
        [key: string]: { value: number | string; bin_size: number };
      } = {};
      Object.keys(currentFeatures).forEach((featureName) => {
        const val = currentFeatures[featureName];
        if (
          typeof val === 'number' &&
          isNaN(currentFeatures[featureName] as number)
        )
          return;
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
        if (
          modelResults.data.length > 0 &&
          !analysis.models.includes(modelResults)
        ) {
          analysis.models.push(modelResults);
        }
        setRangeAnalysis({ ...analysis });
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

      const binSize = binningValues[featureName] ?? 0;
      analysis.datasetReference = datasetResults;

      for (const featureVal of featureValueRange) {
        try {
          const featuresWithBins: {
            [key: string]: { value: number | string; bin_size: number };
          } = {};
          Object.keys(featureValues).forEach((featureName) => {
            const val = featureValues[featureName];
            if (
              typeof val === 'number' &&
              isNaN(featureValues[featureName] as number)
            )
              return;
            featuresWithBins[featureName] = {
              value: featureValues[featureName],
              bin_size: binningValues[featureName] ?? 0,
            };
          });
          // Add the varying feature with its bin
          featuresWithBins[featureName] = {
            value: featureVal,
            bin_size: binSize,
          };

          // Map back to featureValues
          const featureValuesTemp: FeatureValues = {};
          Object.entries(featuresWithBins).forEach(
            ([fname, fval]) => (featureValuesTemp[fname] = fval.value)
          );
          // Apply contraints
          const featureValuesTempContrained =
            applyContraints(featureValuesTemp);
          // Map back to bins
          Object.entries(featureValuesTempContrained).forEach(
            ([featureName, fval]) =>
              (featuresWithBins[featureName] = {
                ...featuresWithBins[featureName],
                value: fval,
              })
          );

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
          if (data.n_samples > 5) {
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
        setRangeAnalysis({ ...analysis });
      }

      if (datasetResults.data.length > 0) {
        analysis.datasetReference = datasetResults;
      }
      setRangeAnalysis({ ...analysis });
    } catch (err) {
      console.warn('Error generating dataset reference:', err);
    }

    return analysis;
  };

  return (
    <div className='app'>
      <nav className='tabs'>
        <button
          className={`tab ${activeTab === 'prediction' ? 'active' : ''}`}
          onClick={() => setActiveTab('prediction')}
        >
          🎯 Win Probability Prediction
        </button>
        <button
          className={`tab ${activeTab === 'demo-analysis' ? 'active' : ''}`}
          onClick={() => setActiveTab('demo-analysis')}
        >
          📊 Demo Analysis
        </button>
      </nav>

      {activeTab === 'prediction' ? (
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

                <div className='range-analysis'>
                  <FeatureRangeAnalyzer
                    features={features}
                    featureValues={featureValues}
                    selectedModels={selectedModels}
                    onAnalyze={handleRangeAnalysis}
                  />

                  <Plot2D
                    analysis={rangeAnalysis}
                    predictions={predictions}
                    featureValues={featureValues}
                    width={800}
                    height={300}
                  />
                </div>
              </>
            )}
          </div>

          {selectedModels.length > 0 && (
            <section className='prediction-section'>
              {error && <div className='error'>{error}</div>}

              <PredictionResults predictions={predictions} />
            </section>
          )}
        </main>
      ) : (
        <main className='main-content'>
          <DemoAnalysis
            demoId={demoId}
            filename={demoFilename}
            players={demoPlayers}
            selectedPlayer={selectedPlayer}
            analysisData={analysisData}
            onDemoUpload={(id, name, playerList) => {
              setDemoId(id);
              setDemoFilename(name);
              setDemoPlayers(playerList);
              setSelectedPlayer(null);
              setAnalysisData(null);
            }}
            onPlayerSelect={(playerName) => {
              setSelectedPlayer(playerName);
              setAnalysisData(null);
            }}
            onAnalysisComplete={(data) => {
              setAnalysisData(data);
            }}
            onError={(errorMsg) => {
              setError(errorMsg);
            }}
          />
        </main>
      )}
    </div>
  );
}

export default App;
