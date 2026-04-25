"""ViralPredictor - Machine learning model for predicting viral potential."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib


class ViralPredictor:
    """
    ML model for predicting social media content virality.
    
    Uses engagement metrics and content features to predict
    whether content will go viral.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the ViralPredictor.
        
        Args:
            model_path: Optional path to load a pre-trained model.
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'followers_count',
            'following_count', 
            'posts_count',
            'has_profile_picture',
            'has_bio',
            'account_age_days',
            'content_length',
            'has_hashtags',
            'has_mentions',
            'has_url',
            'hour_of_day',
            'day_of_week',
            'is_weekend',
            'engagement_rate_history',
            'avg_likes_history',
            'avg_comments_history',
            'follower_to_following_ratio',
            'posts_per_day'
        ]
        self._is_trained = False
        self._feature_importance: Optional[Dict[str, float]] = None
        
        if model_path:
            self.load_model(model_path)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features from raw engagement metrics.
        
        Args:
            df: DataFrame with raw features.
            
        Returns:
            DataFrame with engineered features.
        """
        df = df.copy()
        
        # Ratio features
        df['follower_to_following_ratio'] = np.where(
            df['following_count'] > 0,
            df['followers_count'] / df['following_count'],
            df['followers_count']
        )
        
        # Activity rate
        df['account_age_days'] = np.maximum(df['account_age_days'], 1)
        df['posts_per_day'] = df['posts_count'] / df['account_age_days']
        
        # Binary features
        df['has_profile_picture'] = df['has_profile_picture'].astype(int)
        df['has_bio'] = df['has_bio'].astype(int)
        df['has_hashtags'] = df['has_hashtags'].astype(int)
        df['has_mentions'] = df['has_mentions'].astype(int)
        df['has_url'] = df['has_url'].astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Interaction features
        df['engagement_rate_history'] = np.where(
            df['followers_count'] > 0,
            df['avg_likes_history'] / df['followers_count'],
            0
        )
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input.
        
        Args:
            df: DataFrame with raw features.
            
        Returns:
            Scaled feature array.
        """
        df = self._engineer_features(df)
        
        # Ensure all feature columns exist
        for col in self.feature_names:
            if col not in df.columns:
                if col == 'is_weekend':
                    df[col] = df['day_of_week'].isin([5, 6]).astype(int)
                elif col == 'engagement_rate_history':
                    df[col] = df['avg_likes_history'] / np.maximum(df['followers_count'], 1)
                elif col == 'follower_to_following_ratio':
                    df[col] = df['followers_count'] / np.maximum(df['following_count'], 1)
                elif col == 'posts_per_day':
                    df[col] = df['posts_count'] / np.maximum(df['account_age_days'], 1)
                else:
                    df[col] = 0
        
        X = df[self.feature_names].values
        
        if self._is_trained:
            X = self.scaler.transform(X)
        else:
            X = self.scaler.fit_transform(X)
            
        return X
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Train the viral prediction model.
        
        Args:
            X: Feature matrix.
            y: Target labels (1 = viral, 0 = not viral).
            test_size: Fraction of data for testing.
            cv_folds: Number of cross-validation folds.
            
        Returns:
            Dictionary with training metrics.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self._is_trained = True
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': self._get_feature_importance()
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, 
            np.vstack([X_train_scaled, X_test_scaled]),
            np.hstack([y_train, y_test]),
            cv=cv_folds,
            scoring='roc_auc'
        )
        metrics['cv_roc_auc_mean'] = cv_scores.mean()
        metrics['cv_roc_auc_std'] = cv_scores.std()
        
        return metrics
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        self._feature_importance = dict(zip(self.feature_names, importance))
        return self._feature_importance
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Tuple of (predictions, probabilities).
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def predict_single(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict virality for a single content piece.
        
        Args:
            features: Dictionary of content features.
            
        Returns:
            Dictionary with prediction and confidence.
        """
        df = pd.DataFrame([features])
        X = self.prepare_features(df)
        pred, prob = self.predict(X)
        
        return {
            'will_go_viral': bool(pred[0]),
            'confidence_score': float(prob[0]),
            'viral_probability_percent': round(prob[0] * 100, 2)
        }
    
    def save_model(self, path: str) -> None:
        """Save the trained model to disk.
        
        Args:
            path: Path to save the model.
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self._is_trained,
            'feature_importance': self._feature_importance
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained model from disk.
        
        Args:
            path: Path to the saved model.
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self._is_trained = model_data['is_trained']
        self._feature_importance = model_data.get('feature_importance')
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained
