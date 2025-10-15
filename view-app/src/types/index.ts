// Types for the CS2 Win Probability Predictor

export interface Model {
  display_name: string;
  feature_count: number;
  accuracy: number;
  auc: number;
  description: string;
  name: string;
}

export interface ModelResponse {
  [key: string]: Model;
}

export interface FeatureConstraints {
  type: 'number' | 'checkbox' | 'select';
  min?: number;
  max?: number;
  step?: number;
  options?: [string | number, string][];
}

export interface Feature {
  name: string;
  display_name: string;
  default: number | boolean | string;
  constraints: FeatureConstraints;
}

export interface FeatureCompatibility {
  supportingModels: string[];
  modelCount: number;
  isCommon: boolean;
}

export interface Prediction {
  ct_win_probability: number;
  t_win_probability: number;
  predicted_winner: string;
  confidence: number;
}

export interface ModelInfo {
  name: string;
  accuracy: number;
  auc: number;
  feature_count: number;
}

export interface PredictionResponse {
  success: boolean;
  prediction?: Prediction;
  model_info?: ModelInfo;
  error?: string;
}

export interface PredictionResult {
  modelName: string;
  prediction: Prediction;
  model_info: ModelInfo;
}

export interface FeatureValues {
  [key: string]: number;
}

export interface BinningValues {
  [key: string]: number;
}

export interface ScenarioData {
  [key: string]: number;
}

export interface Scenarios {
  [key: string]: ScenarioData;
}

export interface SliceDatasetResponse {
  ct_win_probability: number;
  t_win_probability: number;
  n_samples: number;
}

export interface PredictionWithBinning extends PredictionResult {
  binningData?: SliceDatasetResponse;
}

// Feature Range Analysis Types
export interface FeatureRangePoint {
  featureValue: number;
  ct_win_probability: number;
  t_win_probability: number;
}

export interface ModelRangeResult {
  modelName: string;
  model_info: ModelInfo;
  data: FeatureRangePoint[];
}

export interface DatasetRangeResult {
  data: FeatureRangePoint[];
  totalSamples: number;
}

export interface FeatureRangeAnalysis {
  featureName: string;
  models: ModelRangeResult[];
  datasetReference?: DatasetRangeResult;
}

// Demo Analysis Types
export interface PlayerAnalysisEvent {
  round: number;
  side: 'ct' | 't';
  impact: number;
  event_round: number;
  event_type:
    | 'kill'
    | 'death'
    | 'death (t)'
    | 'assist'
    | 'flash_assist'
    | 'bomb_plant'
    | string;
  game_state: string;
  pre_win: number;
  post_win: number;
  post_plant: boolean;
  tick: number;
  trade: boolean;
}

export interface AllPlayersAnalysisResponse {
  success: boolean;
  demo_filename?: string;
  analysis?: Record<string, PlayerAnalysisEvent[]>;
  players?: string[];
  error?: string;
}

export interface DemoUploadResponse {
  success: boolean;
  demo_id?: string;
  filename?: string;
  players?: string[];
  player_count?: number;
  error?: string;
}
