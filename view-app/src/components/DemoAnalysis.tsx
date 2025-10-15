import { useState } from 'react';
import { API_ENDPOINTS } from '../config/api';
import type {
  DemoUploadResponse,
  PlayerAnalysisResponse,
  AllPlayersAnalysisResponse,
  PlayerAnalysisEvent,
} from '../types';
import PlayerImpactChart from './PlayerImpactChart';
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
  players,
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
  const [useV2Api, setUseV2Api] = useState(true); // Try v2 first, fallback to v1

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

    // If we already have all players data (v2), use cached data
    if (allPlayersData && allPlayersData[playerName]) {
      onAnalysisComplete(allPlayersData[playerName]);
      return;
    }

    setAnalyzing(true);
    setError(null);

    try {
      // Try v2 API first (all players at once)
      if (useV2Api && !allPlayersData) {
        try {
          const v2Response = await fetch(
            API_ENDPOINTS.allPlayersAnalysis(demoId)
          );
          const v2Data: AllPlayersAnalysisResponse = await v2Response.json();

          if (v2Data.success && v2Data.analysis) {
            // Cache all players data
            setAllPlayersData(v2Data.analysis);

            // Set current player's data
            if (v2Data.analysis[playerName]) {
              onAnalysisComplete(v2Data.analysis[playerName]);
              setError(null);
              setAnalyzing(false);
              return;
            }
          }
        } catch (v2Error) {
          console.warn('V2 API failed, falling back to V1:', v2Error);
          setUseV2Api(false);
        }
      }

      // Fallback to v1 API (single player)
      const response = await fetch(
        API_ENDPOINTS.playerAnalysis(demoId, playerName)
      );
      const data: PlayerAnalysisResponse = await response.json();

      if (data.success && data.analysis) {
        onAnalysisComplete(data.analysis);
        setError(null);
      } else {
        const errorMsg = data.error || 'Failed to analyze player';
        setError(errorMsg);
        onError(errorMsg);
      }
    } catch (err) {
      const errorMsg = 'Error analyzing player: ' + (err as Error).message;
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

      {error && (
        <div
          className='error-message'
          style={{ color: 'red', padding: '10px' }}
        >
          ❌ {error}
        </div>
      )}

      {players.length > 0 && (
        <section className='players-section'>
          <h2>🎮 Players ({players.length})</h2>
          <div className='players-grid'>
            {players.map((player) => (
              <button
                key={player}
                className={`player-button ${
                  selectedPlayer === player ? 'selected' : ''
                }`}
                onClick={() => handlePlayerSelect(player)}
                disabled={analyzing}
              >
                {player}
              </button>
            ))}
          </div>
        </section>
      )}

      {analyzing && (
        <div className='loading-message'>⏳ Analyzing player data...</div>
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
