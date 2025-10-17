import { useState } from 'react';
import { API_ENDPOINTS } from '../config/api';
import type {
  DemoUploadResponse,
  AllPlayersAnalysisResponse,
  PlayerAnalysisEvent,
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

  const handleAutoAnalysis = async (currentDemoId: string) => {
    setAnalyzing(true);
    setError(null);

    try {
      const response = await fetch(
        API_ENDPOINTS.allPlayersAnalysis(currentDemoId)
      );
      const data: AllPlayersAnalysisResponse = await response.json();

      if (data.success && data.analysis) {
        // Cache all players data
        setAllPlayersData(data.analysis);
        setError(null);
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
      const response = await fetch(API_ENDPOINTS.allPlayersAnalysis(demoId));
      const data: AllPlayersAnalysisResponse = await response.json();

      if (data.success && data.analysis) {
        // Cache all players data
        setAllPlayersData(data.analysis);

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
