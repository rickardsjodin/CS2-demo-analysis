# CS2 Win Probability Predictor - React App

A modern React TypeScript application for predicting CS2 match outcomes using machine learning models.

## 🚀 Features

- **Multi-Model Comparison**: Select and compare multiple ML models simultaneously
- **Interactive Feature Inputs**: Range sliders with real-time value display
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Real-time Predictions**: Instant prediction updates as you adjust parameters
- **Quick Scenarios**: Pre-configured game scenarios for rapid testing
- **Modern UI**: Clean, intuitive interface with smooth animations

## 🛠️ Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **CSS Modules** for component-scoped styling
- **Modern CSS** (Grid, Flexbox, CSS Variables)

## 📦 Installation

1. Clone the repository
2. Navigate to the `view-app` directory
3. Install dependencies:

```bash
npm install
```

## 🏃‍♂️ Development

Start the development server:

```bash
npm run dev
```

The application will be available at `http://localhost:5173/`

## 🏗️ Building

Build for production:

```bash
npm run build
```

## 🌐 API Integration

The app expects a Flask backend with the following endpoints:

- `GET /api/models` - Get available ML models
- `GET /api/model/{model_name}/features` - Get features for a specific model
- `POST /api/predict` - Make predictions with feature data

## 📱 Components

### ModelSelection

- Multi-model selection with checkboxes
- Model performance metrics display
- Visual selection feedback

### FeatureInputs

- Dynamic feature input generation
- Range sliders with value display
- Feature compatibility indicators
- Grouped feature organization

### PredictionResults

- Single model prediction display
- Multi-model comparison view
- Progress bars for win probabilities
- Model performance metrics

### ScenarioButtons

- Quick scenario loading
- Pre-configured game states
- Disabled state handling

## 🎮 Usage

1. **Select Models**: Choose one or more ML models to compare
2. **Adjust Features**: Use sliders to set game state parameters
3. **Quick Scenarios**: Click scenario buttons for pre-configured states
4. **View Predictions**: Compare predictions across selected models
5. **Analyze Results**: Review win probabilities and confidence levels

## 🔧 Configuration

The app can be configured by modifying:

- Feature categories in `FeatureInputs.tsx`
- Scenario presets in `ScenarioButtons.tsx`
- Styling variables in CSS files
- API endpoints in component fetch calls

## 📝 Type Definitions

All TypeScript types are defined in `src/types/index.ts` for:

- Model information and responses
- Feature definitions and constraints
- Prediction results and comparisons
- Component prop interfaces

## 🚀 Deployment

The built application can be deployed to any static hosting service:

1. Run `npm run build`
2. Deploy the `dist` folder contents
3. Ensure the backend API is accessible from the deployment domain

## 🧪 Features Showcase

- **Real-time Updates**: Predictions update automatically as you adjust sliders
- **Multi-Model Comparison**: Side-by-side comparison of different ML models
- **Responsive Design**: Optimized for both desktop and mobile viewing
- **Accessibility**: Proper form labels and keyboard navigation support
- **Performance**: Optimized bundle size and fast loading times
