import { useState, useEffect } from 'react';
import { API_ENDPOINTS } from '../config/api';
import type {
  DemoUploadResponse,
  AllPlayersAnalysisResponse,
  PlayerAnalysisEvent,
  Model,
} from '../types';
import PlayerImpactChart from './PlayerImpactChart';
import PlayerOverview from './PlayerOverview';
import './DemoAnalysis.css';

interface DemoAnalysisProps {
  demoId: string | null;
  filename: string | null;
  players: string[];
  selectedPlayer: string | null;
  analysisData: PlayerAnalysisEvent[] | null;
  onDemoUpload: (demoId: string, filename: string, players: string[]) => void;
  onPlayerSelect: (playerName: string) => void;
  onAnalysisComplete: (data: PlayerAnalysisEvent[]) => void;
  onError: (error: string) => void;
}

function DemoAnalysis({
  demoId,
  filename,
  selectedPlayer,
  analysisData,
  onDemoUpload,
  onPlayerSelect,
  onAnalysisComplete,
  onError,
}: DemoAnalysisProps) {
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [allPlayersData, setAllPlayersData] = useState<Record<
    string,
    PlayerAnalysisEvent[]
  > | null>(null);
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('xgboost_hltv');
  const [modelUsed, setModelUsed] = useState<string>('xgboost_hltv');
  const [loadingModels, setLoadingModels] = useState(false);

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      setLoadingModels(true);
      try {
        const response = await fetch(API_ENDPOINTS.models);
        const data = await response.json();
        console.log('Models API response:', data);
        if (data.success && data.models) {
          // Convert array of models to Model[]
          setAvailableModels(data.models);
          console.log('Available models set:', data.models);
          // Set default model if available
          if (data.models.length > 0) {
            setSelectedModel(data.models[0].name);
            setModelUsed(data.models[0].name);
          }
        }
      } catch (err) {
        console.error('Error fetching models:', err);
      } finally {
        setLoadingModels(false);
      }
    };

    fetchModels();
  }, []);

  const handleAutoAnalysis = async (
    currentDemoId: string,
    modelToUse?: string
  ) => {
    setAnalyzing(true);
    setError(null);

    const modelName = modelToUse || selectedModel;

    try {
      const response = await fetch(
        API_ENDPOINTS.allPlayersAnalysis(currentDemoId, modelName)
      );
      const data: AllPlayersAnalysisResponse = await response.json();

      if (data.success && data.analysis) {
        // Cache all players data
        setAllPlayersData(data.analysis);
        setError(null);
        // Update the model that was actually used
        if (data.model_used) {
          setModelUsed(data.model_used);
        }
      } else {
        const errorMsg = data.error || 'Failed to analyze players';
        setError(errorMsg);
        onError(errorMsg);
      }
    } catch (err) {
      const errorMsg = 'Error analyzing players: ' + (err as Error).message;
      setError(errorMsg);
      onError(errorMsg);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleModelChange = (newModel: string) => {
    setSelectedModel(newModel);
    // Clear cached analysis when model changes
    setAllPlayersData(null);
    onAnalysisComplete([]);

    // Re-run analysis with new model if we have a demo
    // Pass the new model directly to avoid stale state
    if (demoId) {
      handleAutoAnalysis(demoId, newModel);
    }
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Check file extension
    if (!file.name.endsWith('.dem')) {
      setError('Please upload a .dem file');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(API_ENDPOINTS.demoUpload, {
        method: 'POST',
        body: formData,
      });

      const data: DemoUploadResponse = await response.json();

      if (data.success && data.demo_id && data.players) {
        onDemoUpload(data.demo_id, data.filename || file.name, data.players);
        setError(null);

        // Automatically trigger analysis after successful upload
        await handleAutoAnalysis(data.demo_id);
      } else {
        const errorMsg = data.error || 'Failed to upload demo';
        setError(errorMsg);
        onError(errorMsg);
      }
    } catch (err) {
      const errorMsg = 'Error uploading file: ' + (err as Error).message;
      setError(errorMsg);
      onError(errorMsg);
    } finally {
      setUploading(false);
    }
  };

  const handlePlayerSelect = async (playerName: string) => {
    if (!demoId) return;

    onPlayerSelect(playerName);

    // If we already have all players data, use cached data
    if (allPlayersData && allPlayersData[playerName]) {
      onAnalysisComplete(allPlayersData[playerName]);
      return;
    }

    setAnalyzing(true);
    setError(null);

    try {
      const response = await fetch(
        API_ENDPOINTS.allPlayersAnalysis(demoId, selectedModel)
      );
      const data: AllPlayersAnalysisResponse = await response.json();

      if (data.success && data.analysis) {
        // Cache all players data
        setAllPlayersData(data.analysis);

        // Update the model that was actually used
        if (data.model_used) {
          setModelUsed(data.model_used);
        }

        // Set current player's data
        if (data.analysis[playerName]) {
          onAnalysisComplete(data.analysis[playerName]);
          setError(null);
        } else {
          const errorMsg = `No data found for player ${playerName}`;
          setError(errorMsg);
          onError(errorMsg);
        }
      } else {
        const errorMsg = data.error || 'Failed to analyze players';
        setError(errorMsg);
        onError(errorMsg);
      }
    } catch (err) {
      const errorMsg = 'Error analyzing players: ' + (err as Error).message;
      setError(errorMsg);
      onError(errorMsg);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className='demo-analysis'>
      <section className='upload-section'>
        <h2>📁 Upload Demo File</h2>
        <div className='upload-area'>
          <input
            type='file'
            accept='.dem'
            onChange={handleFileUpload}
            disabled={uploading}
            id='demo-upload'
            style={{ display: 'none' }}
          />
          <label htmlFor='demo-upload' className='upload-button'>
            {uploading ? '⏳ Uploading...' : '📤 Choose .dem file'}
          </label>
          {filename && <p className='filename'>📄 {filename}</p>}
        </div>
      </section>

      {demoId && !loadingModels && availableModels.length > 0 && (
        <section className='model-selection-section'>
          <h3>🤖 Prediction Model</h3>
          <div className='demo-model-selector'>
            <select
              value={selectedModel}
              onChange={(e) => handleModelChange(e.target.value)}
              disabled={analyzing}
              className='model-dropdown'
            >
              {availableModels.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.display_name} (AUC: {model.auc.toFixed(3)})
                </option>
              ))}
            </select>
            {allPlayersData && modelUsed && modelUsed !== selectedModel && (
              <p className='model-info'>
                ℹ️ Results are from model: {modelUsed}. Change selection above
                to re-analyze.
              </p>
            )}
          </div>
        </section>
      )}

      {demoId && loadingModels && (
        <section className='model-selection-section'>
          <h3>🤖 Prediction Model</h3>
          <div className='loading-message'>Loading models...</div>
        </section>
      )}

      {error && (
        <div
          className='error-message'
          style={{ color: 'red', padding: '10px' }}
        >
          ❌ {error}
        </div>
      )}

      {analyzing && (
        <div className='loading-message'>⏳ Analyzing player data...</div>
      )}

      {allPlayersData && (
        <PlayerOverview
          allPlayersData={allPlayersData}
          selectedPlayer={selectedPlayer}
          onPlayerSelect={handlePlayerSelect}
          analyzing={analyzing}
        />
      )}

      {analysisData && selectedPlayer && (
        <section className='analysis-section'>
          <h2>📊 Impact Analysis: {selectedPlayer}</h2>
          <PlayerImpactChart data={analysisData} playerName={selectedPlayer} />
        </section>
      )}
    </div>
  );
}

export default DemoAnalysis;
