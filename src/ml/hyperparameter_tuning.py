"""
Hyperparameter tuning module for CS2 win probability models.
Provides systematic hyperparameter optimization using GridSearchCV and RandomizedSearchCV.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, log_loss
from typing import Dict, Any, Optional, Tuple, Union
import warnings
from pathlib import Path
import sys
import json
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Hyperparameter tuning class for CS2 win probability models.
    Supports multiple search strategies and model types.
    """
    
    def __init__(self, scoring='roc_auc', cv_folds=5, n_jobs=-1, verbose=1):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            scoring: Scoring metric for optimization ('roc_auc', 'accuracy', 'neg_log_loss')
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Verbosity level
        """
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Set up cross-validation strategy
        self.cv_strategy = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=42
        )
        
        # Custom scorer for AUC (handles edge cases)
        self.auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    
    def tune_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        param_grid: Optional[Dict[str, list]] = None,
        search_method: str = 'grid',
        n_iter: int = 50
    ) -> Tuple[object, Dict[str, Any], float]:
        """
        Tune XGBoost hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            param_grid: Custom parameter grid (uses default if None)
            search_method: 'grid' or 'random'
            n_iter: Number of iterations for random search
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        # Default parameter grid for XGBoost
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.5],
                'min_child_weight': [1, 3, 5, 7],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0]
            }
        
        # Base model
        base_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=1  # Set to 1 to avoid conflicts with GridSearchCV n_jobs
        )
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=self.scoring,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=42,
                return_train_score=True
            )
        
        print(f"🔍 Starting XGBoost {search_method} search with {len(param_grid)} parameters...")
        print(f"   Total combinations: {self._estimate_combinations(param_grid, search_method, n_iter)}")
        
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def tune_lightgbm(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        param_grid: Optional[Dict[str, list]] = None,
        search_method: str = 'grid',
        n_iter: int = 50
    ) -> Tuple[object, Dict[str, Any], float]:
        """
        Tune LightGBM hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            param_grid: Custom parameter grid (uses default if None)
            search_method: 'grid' or 'random'
            n_iter: Number of iterations for random search
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        
        # Default parameter grid for LightGBM
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'min_child_samples': [5, 10, 20, 30],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0],
                'num_leaves': [15, 31, 63, 127]
            }
        
        # Base model
        base_model = lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            n_jobs=1  # Set to 1 to avoid conflicts with GridSearchCV n_jobs
        )
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=self.scoring,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=42,
                return_train_score=True
            )
        
        print(f"🔍 Starting LightGBM {search_method} search with {len(param_grid)} parameters...")
        print(f"   Total combinations: {self._estimate_combinations(param_grid, search_method, n_iter)}")
        
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def tune_random_forest(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        param_grid: Optional[Dict[str, list]] = None,
        search_method: str = 'grid',
        n_iter: int = 50
    ) -> Tuple[object, Dict[str, Any], float]:
        """
        Tune Random Forest hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            param_grid: Custom parameter grid (uses default if None)
            search_method: 'grid' or 'random'
            n_iter: Number of iterations for random search
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Default parameter grid for Random Forest
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
        
        # Base model
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=1  # Set to 1 to avoid conflicts with GridSearchCV n_jobs
        )
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=self.scoring,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=42,
                return_train_score=True
            )
        
        print(f"🔍 Starting Random Forest {search_method} search with {len(param_grid)} parameters...")
        print(f"   Total combinations: {self._estimate_combinations(param_grid, search_method, n_iter)}")
        
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def tune_logistic_regression(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        param_grid: Optional[Dict[str, list]] = None,
        search_method: str = 'grid',
        n_iter: int = 50
    ) -> Tuple[object, Dict[str, Any], float]:
        """
        Tune Logistic Regression hyperparameters.
        
        Args:
            X_train: Training features (should be pre-scaled)
            y_train: Training targets
            param_grid: Custom parameter grid (uses default if None)
            search_method: 'grid' or 'random'
            n_iter: Number of iterations for random search
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        from sklearn.linear_model import LogisticRegression
        
        # Default parameter grid for Logistic Regression
        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga', 'lbfgs'],
                'max_iter': [500, 1000, 2000],
                'class_weight': ['balanced', None]
            }
        
        # Base model
        base_model = LogisticRegression(
            random_state=42,
            n_jobs=1  # Set to 1 to avoid conflicts with GridSearchCV n_jobs
        )
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=self.scoring,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=42,
                return_train_score=True
            )
        
        print(f"🔍 Starting Logistic Regression {search_method} search with {len(param_grid)} parameters...")
        print(f"   Total combinations: {self._estimate_combinations(param_grid, search_method, n_iter)}")
        
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def tune_neural_network(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        param_grid: Optional[Dict[str, list]] = None,
        search_method: str = 'grid',
        n_iter: int = 50
    ) -> Tuple[object, Dict[str, Any], float]:
        """
        Tune Neural Network (MLPClassifier) hyperparameters.
        
        Args:
            X_train: Training features (should be pre-scaled)
            y_train: Training targets
            param_grid: Custom parameter grid (uses default if None)
            search_method: 'grid' or 'random'
            n_iter: Number of iterations for random search
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        from sklearn.neural_network import MLPClassifier
        
        # Default parameter grid for Neural Network
        if param_grid is None:
            param_grid = {
                'hidden_layer_sizes': [
                    (50,), (100,), (50, 50), (100, 50), (100, 100),
                    (50, 30, 20), (100, 50, 25)
                ],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [500, 1000, 2000]
            }
        
        # Base model
        base_model = MLPClassifier(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=self.scoring,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=42,
                return_train_score=True
            )
        
        print(f"🔍 Starting Neural Network {search_method} search with {len(param_grid)} parameters...")
        print(f"   Total combinations: {self._estimate_combinations(param_grid, search_method, n_iter)}")
        
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def _estimate_combinations(self, param_grid: Dict[str, list], search_method: str, n_iter: int) -> str:
        """Estimate the number of parameter combinations to be tested."""
        if search_method == 'grid':
            total = 1
            for values in param_grid.values():
                total *= len(values)
            return f"{total:,} (Grid Search)"
        else:
            return f"{n_iter:,} (Random Search)"
    
    def save_tuning_results(self, model_name: str, best_params: Dict[str, Any], 
                          best_score: float, results_dir: Path) -> None:
        """
        Save hyperparameter tuning results to file.
        
        Args:
            model_name: Name of the model
            best_params: Best parameters found
            best_score: Best score achieved
            results_dir: Directory to save results
        """
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'model_name': model_name,
            'best_params': best_params,
            'best_score': best_score,
            'scoring_metric': self.scoring,
            'cv_folds': self.cv_folds,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        filename = results_dir / f"{model_name}_hyperparameter_tuning_results.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Hyperparameter tuning results saved to: {filename}")


def get_model_tuner(model_type: str) -> callable:
    """
    Get the appropriate tuning method for a model type.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', etc.)
        
    Returns:
        Tuning method function
    """
    tuner = HyperparameterTuner()
    
    tuning_methods = {
        'xgboost': tuner.tune_xgboost,
        'xgboost_hltv': tuner.tune_xgboost,
        'xgboost_hltv_time': tuner.tune_xgboost,
        'lightgbm': tuner.tune_lightgbm,
        'random_forest': tuner.tune_random_forest,
        'logistic_regression': tuner.tune_logistic_regression,
        'neural_network': tuner.tune_neural_network
    }
    
    return tuning_methods.get(model_type)


def quick_tune_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                    search_method: str = 'random', n_iter: int = 20) -> Tuple[object, Dict[str, Any], float]:
    """
    Quick hyperparameter tuning for a specific model type.
    
    Args:
        model_type: Type of model to tune
        X_train: Training features
        y_train: Training targets
        search_method: 'grid' or 'random'
        n_iter: Number of iterations for random search
        
    Returns:
        Tuple of (best_model, best_params, best_score)
    """
    tuning_method = get_model_tuner(model_type)
    
    if tuning_method is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"🎯 Quick tuning {model_type} with {search_method} search...")
    
    return tuning_method(
        X_train=X_train,
        y_train=y_train,
        search_method=search_method,
        n_iter=n_iter
    )


# Predefined parameter grids for different complexity levels
PARAM_GRIDS = {
    'xgboost_quick': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'xgboost_thorough': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    },
    'lightgbm_quick': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'random_forest_quick': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.8]
    }
}


def get_param_grid(model_type: str, complexity: str = 'quick') -> Dict[str, list]:
    """
    Get predefined parameter grid for a model type and complexity level.
    
    Args:
        model_type: Type of model
        complexity: 'quick' or 'thorough'
        
    Returns:
        Parameter grid dictionary
    """
    grid_key = f"{model_type}_{complexity}"
    return PARAM_GRIDS.get(grid_key, PARAM_GRIDS.get(f"{model_type}_quick"))