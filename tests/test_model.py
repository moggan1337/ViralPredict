"""Pytest tests for ViralPredict model."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.model import ViralPredictor
from src.data import generate_synthetic_data, get_balanced_dataset
from src.predict import predict_virality, predict_batch, get_sample_features


class TestViralPredictor:
    """Test suite for ViralPredictor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        X, y = generate_synthetic_data(n_samples=500, viral_ratio=0.3, seed=42)
        return X, y
    
    @pytest.fixture
    def balanced_data(self):
        """Generate balanced training data."""
        return get_balanced_dataset(n_samples=200, seed=42)
    
    @pytest.fixture
    def trained_predictor(self, balanced_data):
        """Create and train a predictor."""
        X, y = balanced_data
        predictor = ViralPredictor()
        predictor.train(X.values, y.values)
        return predictor
    
    def test_initialization(self):
        """Test model initializes correctly."""
        predictor = ViralPredictor()
        assert predictor is not None
        assert not predictor.is_trained
        assert len(predictor.feature_names) == 18
    
    def test_feature_engineering(self, sample_data):
        """Test feature engineering creates all features."""
        X, _ = sample_data
        predictor = ViralPredictor()
        X_eng = predictor._engineer_features(X)
        
        assert 'follower_to_following_ratio' in X_eng.columns
        assert 'posts_per_day' in X_eng.columns
        assert 'is_weekend' in X_eng.columns
        assert 'engagement_rate_history' in X_eng.columns
    
    def test_prepare_features(self, sample_data):
        """Test feature preparation returns correct shape."""
        X, _ = sample_data
        predictor = ViralPredictor()
        X_prepared = predictor.prepare_features(X)
        
        assert X_prepared.shape[0] == X.shape[0]
        assert X_prepared.shape[1] == len(predictor.feature_names)
    
    def test_train_model(self, balanced_data):
        """Test model training produces trained model."""
        X, y = balanced_data
        predictor = ViralPredictor()
        metrics = predictor.train(X.values, y.values)
        
        assert predictor.is_trained
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_train_metrics_reasonable(self, balanced_data):
        """Test trained model achieves reasonable accuracy."""
        X, y = balanced_data
        predictor = ViralPredictor()
        predictor.train(X.values, y.values, test_size=0.2)
        
        # On balanced data with clear patterns, should get >60% accuracy
        assert metrics['accuracy'] > 0.6 if 'metrics' else True
    
    def test_predict_single(self, trained_predictor):
        """Test single prediction works."""
        features = get_sample_features(
            followers_count=50000,
            following_count=200,
            posts_count=1000,
            content_length=300,
            has_hashtags=True,
            has_mentions=True,
            has_url=True,
            hour_of_day=18,
            day_of_week=5,
            avg_likes_history=5000,
            avg_comments_history=500,
            has_profile_picture=True,
            has_bio=True,
            account_age_days=730
        )
        
        result = trained_predictor.predict_single(features)
        
        assert 'will_go_viral' in result
        assert 'confidence_score' in result
        assert 'viral_probability_percent' in result
        assert isinstance(result['will_go_viral'], bool)
        assert 0 <= result['confidence_score'] <= 1
    
    def test_predict_batch(self, trained_predictor):
        """Test batch prediction works."""
        features_list = [
            get_sample_features(followers_count=50000, avg_likes_history=5000),
            get_sample_features(followers_count=100, avg_likes_history=5),
        ]
        
        predictions = predict_batch(features_list, trained_predictor)
        
        assert len(predictions) == 2
        assert all('will_go_viral' in p for p in predictions)
    
    def test_save_and_load_model(self, trained_predictor):
        """Test model can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.joblib')
            
            trained_predictor.save_model(model_path)
            assert Path(model_path).exists()
            
            new_predictor = ViralPredictor(model_path=model_path)
            assert new_predictor.is_trained
    
    def test_predict_after_load(self, trained_predictor):
        """Test predictions work after loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.joblib')
            trained_predictor.save_model(model_path)
            
            loaded_predictor = ViralPredictor(model_path=model_path)
            features = get_sample_features(followers_count=10000)
            
            original_result = trained_predictor.predict_single(features)
            loaded_result = loaded_predictor.predict_single(features)
            
            assert original_result['confidence_score'] == loaded_result['confidence_score']
    
    def test_untrained_model_raises_error(self):
        """Test that untrained model raises error on predict."""
        predictor = ViralPredictor()
        X = np.random.rand(10, 18)
        
        with pytest.raises(ValueError, match="must be trained"):
            predictor.predict(X)
    
    def test_feature_importance(self, trained_predictor):
        """Test feature importance is calculated."""
        importance = trained_predictor._get_feature_importance()
        
        assert importance is not None
        assert len(importance) == len(trained_predictor.feature_names)
        assert all(0 <= v <= 1 for v in importance.values())
    
    def test_cross_validation(self, balanced_data):
        """Test cross-validation works."""
        X, y = balanced_data
        predictor = ViralPredictor()
        metrics = predictor.train(X.values, y.values, cv_folds=3)
        
        assert 'cv_roc_auc_mean' in metrics
        assert 'cv_roc_auc_std' in metrics
        assert 0 <= metrics['cv_roc_auc_mean'] <= 1


class TestDataGeneration:
    """Test suite for data generation functions."""
    
    def test_generate_synthetic_data_shape(self):
        """Test synthetic data has correct shape."""
        X, y = generate_synthetic_data(n_samples=1000)
        
        assert X.shape[0] == 1000
        assert len(y) == 1000
        assert X.shape[1] == 14  # 14 raw features
    
    def test_generate_synthetic_data_viral_ratio(self):
        """Test viral ratio is approximately correct."""
        X, y = generate_synthetic_data(n_samples=1000, viral_ratio=0.3)
        
        actual_ratio = y.mean()
        assert 0.25 <= actual_ratio <= 0.35
    
    def test_balanced_dataset(self):
        """Test balanced dataset has equal classes."""
        X, y = get_balanced_dataset(n_samples=1000)
        
        assert y.sum() == 500
        assert (1 - y).sum() == 500
    
    def test_feature_ranges(self):
        """Test generated features are in reasonable ranges."""
        X, y = generate_synthetic_data(n_samples=100, seed=42)
        
        assert (X['followers_count'] >= 0).all()
        assert (X['following_count'] >= 0).all()
        assert (X['posts_count'] >= 0).all()
        assert (X['content_length'] >= 0).all()
        assert (0 <= X['hour_of_day']).all()
        assert (X['hour_of_day'] < 24).all()
        assert (0 <= X['day_of_week']).all()
        assert (X['day_of_week'] < 7).all()


class TestPredictFunctions:
    """Test suite for prediction utility functions."""
    
    def test_get_sample_features(self):
        """Test sample features generator."""
        features = get_sample_features(
            followers_count=10000,
            following_count=500
        )
        
        assert features['followers_count'] == 10000
        assert features['following_count'] == 500
        assert 'hour_of_day' in features
        assert 'day_of_week' in features
    
    def test_predict_virality_with_model(self, trained_predictor):
        """Test predict_virality function."""
        features = get_sample_features(followers_count=50000)
        result = predict_virality(features, model=trained_predictor)
        
        assert 'will_go_viral' in result
        assert 'confidence_score' in result
        assert 'confidence_level' in result
        assert 'verdict' in result
    
    def test_predict_virality_with_model_path(self, trained_predictor):
        """Test predict_virality with model path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.joblib')
            trained_predictor.save_model(model_path)
            
            features = get_sample_features()
            result = predict_virality(features, model_path=model_path)
            
            assert 'will_go_viral' in result
    
    def test_predict_virality_requires_model(self):
        """Test predict_virality raises error without model."""
        features = get_sample_features()
        
        with pytest.raises(ValueError, match="model or model_path"):
            predict_virality(features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
