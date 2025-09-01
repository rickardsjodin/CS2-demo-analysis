# CS2 Demo Analysis Toolkit

A comprehensive Python toolkit for analyzing Counter-Strike 2 demo files with machine learning capabilities for win probability prediction and player performance analysis.

## Features

- **Demo Analysis**: Extract detailed player statistics, kill/death analysis, and round-by-round performance metrics
- **Win Probability Prediction**: Machine learning models to predict round win probability based on game state
- **Player Impact Analysis**: Calculate individual player impact on round outcomes using win probability swings
- **Data Visualization**: Generate comprehensive charts and reports for player and team performance
- **Demo Fetching**: Automated tools to download demos from HLTV
- **Caching System**: Efficient caching of parsed demo data for faster subsequent analysis

## Project Structure

```
CS2-demo-analysis/
├── src/                          # Source code
│   ├── core/                     # Core analysis modules
│   │   ├── analysis.py           # Player and round analysis functions
│   │   ├── win_probability.py    # Win probability calculations
│   │   └── snapshot_extractor.py # Game state snapshot extraction
│   ├── utils/                    # Utility modules
│   │   ├── cache_utils.py        # Demo caching utilities
│   │   ├── formatting.py         # Output formatting functions
│   │   └── plotting.py           # Visualization functions
│   ├── ml/                       # Machine learning modules
│   │   ├── train_win_probability_model.py  # Model training
│   │   └── test_win_probability_scenarios.py  # Model testing
│   └── data_fetching/            # Data fetching scripts
│       ├── fetch_hltv_demos.py   # HLTV demo downloader
│       ├── unzip_demos.py        # Demo extraction utilities
│       └── process_demo_directory.py  # Batch demo processing
├── data/                         # Data files
│   ├── models/                   # Trained ML models (.pkl files)
│   └── datasets/                 # JSON datasets and CSV files
├── demos/                        # Demo files (.dem)
├── outputs/                      # Generated reports and visualizations
│   ├── visualizations/           # PNG charts and graphs
│   └── reports/                  # HTML reports
├── docs/                         # Documentation
├── tests/                        # Test files
├── cache/                        # Cached demo data
└── main.py                       # Main entry point
```

## Quick Start

1. **Configure the analysis** by editing the configuration section in `main.py`:

   ```python
   DEMO_FILE = "path/to/your/demo.dem"
   PLAYER_TO_ANALYZE = "PlayerName"
   ```

2. **Run the analysis**:

   ```bash
   python main.py
   ```

3. **View results** in the terminal and check the `outputs/` directory for generated visualizations and reports.

## Key Features

### Player Impact Analysis

- Calculate individual player impact on round outcomes
- Generate round-by-round performance breakdowns
- Compare players side-by-side with impact metrics

### Win Probability Modeling

- Train machine learning models on game state snapshots
- Predict round win probability based on player positions, economy, and game state
- Test models against various scenarios

### Data Visualization

- Player performance charts
- Win probability heatmaps
- Round swing analysis
- Team comparison visualizations

### Demo Processing

- Automated demo downloading from HLTV
- Batch processing of multiple demos
- Efficient caching system for faster re-analysis

## Dependencies

- `awpy` - CS2 demo parsing
- `pandas` / `polars` - Data manipulation
- `scikit-learn` - Machine learning
- `matplotlib` / `seaborn` - Visualization
- `requests` - HTTP requests for demo downloading

## Configuration

The main configuration is in `main.py`. Key settings include:

- `DEMO_FILE`: Path to the demo file to analyze
- `PLAYER_TO_ANALYZE`: Player name for detailed analysis
- `USE_CACHE`: Enable/disable demo caching
- `COMPARE_PLAYER1/2`: Players for comparison analysis

## Machine Learning Models

The toolkit includes several trained models for win probability prediction:

- **Simple CT Win Model**: Basic logistic regression model
- **CT Win Probability Model**: Random Forest model with game state features
- **Improved CT Win Probability Model**: Enhanced model with additional features

Models are stored in `data/models/` and can be retrained using the scripts in `src/ml/`.

## Contributing

This is an analysis toolkit for CS2 demos with a focus on player impact analysis and win probability prediction. The codebase is organized for maintainability and extensibility.

## License

This project is for educational and analysis purposes.
