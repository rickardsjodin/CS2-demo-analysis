// API Configuration
// In development, use Vite proxy. In production, use environment variable or fallback to localhost:5000
const isDevelopment = import.meta.env.DEV;
export const API_BASE_URL = isDevelopment
  ? '' // Use Vite proxy in development
  : import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

// API endpoints
export const API_ENDPOINTS = {
  models: `${API_BASE_URL}/api/models`,
  modelFeatures: (modelName: string) =>
    `${API_BASE_URL}/api/model/${modelName}/features`,
  predict: `${API_BASE_URL}/api/predict`,
} as const;
