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
