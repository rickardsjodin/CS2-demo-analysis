import { useState } from 'react';
import { API_ENDPOINTS } from '../config/api';
import type {
  DemoUploadResponse,
  PlayerAnalysisResponse,
  PlayerAnalysisEvent,
} from '../types';
import PlayerImpactChart from './PlayerImpactChart';
import './DemoAnalysis.css';

function DemoAnalysis() {
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demoId, setDemoId] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [players, setPlayers] = useState<string[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<
    PlayerAnalysisEvent[] | null
  >(null);

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
    setDemoId(null);
    setPlayers([]);
    setSelectedPlayer(null);
    setAnalysisData(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(API_ENDPOINTS.demoUpload, {
        method: 'POST',
        body: formData,
      });

      const data: DemoUploadResponse = await response.json();

      if (data.success && data.demo_id && data.players) {
        setDemoId(data.demo_id);
        setFilename(data.filename || file.name);
        setPlayers(data.players);
      } else {
        setError(data.error || 'Failed to upload demo');
      }
    } catch (err) {
      setError('Error uploading file: ' + (err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const handlePlayerSelect = async (playerName: string) => {
    if (!demoId) return;

    setSelectedPlayer(playerName);
    setAnalyzing(true);
    setError(null);
    setAnalysisData(null);

    try {
      const response = await fetch(
        API_ENDPOINTS.playerAnalysis(demoId, playerName)
      );
      const data: PlayerAnalysisResponse = await response.json();

      if (data.success && data.analysis) {
        setAnalysisData(data.analysis);
      } else {
        setError(data.error || 'Failed to analyze player');
      }
    } catch (err) {
      setError('Error analyzing player: ' + (err as Error).message);
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
