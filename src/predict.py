"""Prediction pipeline for ViralPredict."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from src.model import ViralPredictor


def predict_virality(
    features: Dict[str, Any],
    model: Optional[ViralPredictor] = None,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """Predict if content will go viral.
    
    Args:
        features: Dictionary containing content and user features.
        model: Pre-loaded ViralPredictor instance.
        model_path: Path to a saved model file.
        
    Returns:
        Dictionary with prediction results.
    """
    if model is None:
        if model_path:
            model = ViralPredictor(model_path=model_path)
        else:
            raise ValueError("Either model or model_path must be provided")
    
    result = model.predict_single(features)
    
    # Add interpretation
    if result['confidence_score'] >= 0.7:
        confidence_level = "high"
    elif result['confidence_score'] >= 0.4:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    result['confidence_level'] = confidence_level
    
    if result['will_go_viral']:
        result['verdict'] = "LIKELY TO GO VIRAL"
    else:
        result['verdict'] = "UNLIKELY TO GO VIRAL"
    
    return result


def predict_batch(
    features_list: List[Dict[str, Any]],
    model: ViralPredictor
) -> List[Dict[str, Any]]:
    """Predict virality for multiple content pieces.
    
    Args:
        features_list: List of feature dictionaries.
        model: Trained ViralPredictor instance.
        
    Returns:
        List of prediction results.
    """
    results = []
    
    for features in features_list:
        result = predict_virality(features, model=model)
        results.append(result)
    
    return results


def get_sample_features(
    followers_count: int = 10000,
    following_count: int = 500,
    posts_count: int = 500,
    content_length: int = 200,
    has_hashtags: bool = True,
    has_mentions: bool = False,
    has_url: bool = True,
    hour_of_day: int = 12,
    day_of_week: int = 3,
    avg_likes_history: int = 500,
    avg_comments_history: int = 50,
    has_profile_picture: bool = True,
    has_bio: bool = True,
    account_age_days: int = 365
) -> Dict[str, Any]:
    """Generate sample features for testing.
    
    Args:
        followers_count: Number of followers.
        following_count: Number of accounts being followed.
        posts_count: Total number of posts.
        content_length: Length of content in characters.
        has_hashtags: Whether content has hashtags.
        has_mentions: Whether content has mentions.
        has_url: Whether content includes a URL.
        hour_of_day: Hour of day (0-23).
        day_of_week: Day of week (0=Monday, 6=Sunday).
        avg_likes_history: Average likes on previous posts.
        avg_comments_history: Average comments on previous posts.
        has_profile_picture: Whether user has a profile picture.
        has_bio: Whether user has a bio.
        account_age_days: Age of account in days.
        
    Returns:
        Dictionary of features.
    """
    return {
        'followers_count': followers_count,
        'following_count': following_count,
        'posts_count': posts_count,
        'content_length': content_length,
        'has_hashtags': has_hashtags,
        'has_mentions': has_mentions,
        'has_url': has_url,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'avg_likes_history': avg_likes_history,
        'avg_comments_history': avg_comments_history,
        'has_profile_picture': has_profile_picture,
        'has_bio': has_bio,
        'account_age_days': account_age_days
    }


def explain_prediction(
    result: Dict[str, Any],
    model: ViralPredictor
) -> List[str]:
    """Generate human-readable explanation of prediction.
    
    Args:
        result: Prediction result dictionary.
        model: Trained model for feature importance access.
        
    Returns:
        List of explanation strings.
    """
    explanations = []
    
    if result['will_go_viral']:
        explanations.append(
            f"Prediction: Content is likely to go viral "
            f"(confidence: {result['confidence_score']:.1%})"
        )
    else:
        explanations.append(
            f"Prediction: Content is unlikely to go viral "
            f"(confidence: {1 - result['confidence_score']:.1%})"
        )
    
    if model._feature_importance:
        top_features = sorted(
            model._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        explanations.append("\nTop factors influencing prediction:")
        for feat, importance in top_features:
            explanations.append(f"  - {feat}: {importance:.3f}")
    
    return explanations


if __name__ == "__main__":
    # Example usage
    model = ViralPredictor()
    
    # Sample high-viral-potential content
    viral_features = get_sample_features(
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
    
    # Sample low-viral-potential content
    non_viral_features = get_sample_features(
        followers_count=100,
        following_count=500,
        posts_count=10,
        content_length=50,
        has_hashtags=False,
        has_mentions=False,
        has_url=False,
        hour_of_day=3,
        day_of_week=1,
        avg_likes_history=5,
        avg_comments_history=1,
        has_profile_picture=False,
        has_bio=False,
        account_age_days=7
    )
    
    print("Note: Model needs to be trained before making predictions.")
    print(f"\nSample viral features: {viral_features}")
    print(f"\nSample non-viral features: {non_viral_features}")
