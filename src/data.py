"""Synthetic data generator for training ViralPredict model."""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_synthetic_data(
    n_samples: int = 10000,
    viral_ratio: float = 0.25,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic social media data for training.
    
    This creates realistic-looking engagement data with known
    patterns that the model can learn from.
    
    Args:
        n_samples: Number of samples to generate.
        viral_ratio: Proportion of viral samples (0-1).
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (features DataFrame, target Series).
    """
    np.random.seed(seed)
    
    # Initialize arrays
    data = {}
    
    # User metrics (influence viral potential)
    data['followers_count'] = np.random.lognormal(8, 2, n_samples).astype(int)
    data['following_count'] = np.random.lognormal(6, 1.5, n_samples).astype(int)
    data['posts_count'] = np.random.lognormal(5, 2, n_samples).astype(int)
    data['account_age_days'] = np.random.lognormal(6, 1.5, n_samples).astype(int)
    
    # Binary profile features
    data['has_profile_picture'] = np.random.binomial(1, 0.85, n_samples)
    data['has_bio'] = np.random.binomial(1, 0.7, n_samples)
    
    # Content features
    data['content_length'] = np.random.lognormal(5, 0.8, n_samples).astype(int)
    data['has_hashtags'] = np.random.binomial(1, 0.5, n_samples)
    data['has_mentions'] = np.random.binomial(1, 0.3, n_samples)
    data['has_url'] = np.random.binomial(1, 0.4, n_samples)
    
    # Timing features
    data['hour_of_day'] = np.random.randint(0, 24, n_samples)
    data['day_of_week'] = np.random.randint(0, 7, n_samples)
    
    # Historical engagement
    data['avg_likes_history'] = np.random.lognormal(
        np.log(data['followers_count'] * 0.05),
        1.5,
        n_samples
    ).astype(int)
    data['avg_comments_history'] = (data['avg_likes_history'] * 0.1).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate viral labels based on features
    # Higher viral probability for:
    # - More followers
    # - Higher engagement rate
    # - More posts (active account)
    # - Better account age
    # - Optimal posting times (18-21h, weekends)
    
    viral_score = (
        np.log1p(df['followers_count']) * 0.3 +
        (df['avg_likes_history'] / (df['followers_count'] + 1)) * 10 +
        np.log1p(df['posts_count']) * 0.2 +
        np.log1p(df['account_age_days']) * 0.1 +
        df['has_profile_picture'] * 0.1 +
        df['has_bio'] * 0.05 +
        df['has_hashtags'] * 0.15 +
        ((df['hour_of_day'] >= 17) & (df['hour_of_day'] <= 21)).astype(float) * 0.2 +
        df['day_of_week'].isin([5, 6]).astype(float) * 0.15
    )
    
    # Convert to probability and add noise
    viral_prob = 1 / (1 + np.exp(-(viral_score - 5)))
    viral_prob = np.clip(viral_prob + np.random.normal(0, 0.1, n_samples), 0, 1)
    
    # Generate labels
    y = (np.random.random(n_samples) < viral_prob).astype(int)
    
    # Ensure we have the requested viral ratio
    current_ratio = y.mean()
    if abs(current_ratio - viral_ratio) > 0.05:
        n_viral = int(n_samples * viral_ratio)
        viral_indices = np.where(y == 1)[0]
        non_viral_indices = np.where(y == 0)[0]
        
        if current_ratio > viral_ratio:
            # Too many viral, convert some
            remove_idx = np.random.choice(viral_indices, 
                                          int(len(viral_indices) * viral_ratio / current_ratio),
                                          replace=False)
            y[remove_idx] = 0
        else:
            # Too few viral, convert some
            add_idx = np.random.choice(non_viral_indices,
                                       int(len(non_viral_indices) * current_ratio / (1 - viral_ratio)),
                                       replace=False)
            y[add_idx] = 1
    
    return df, pd.Series(y, name='is_viral')


def generate_viral_examples(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate specifically viral content examples.
    
    Args:
        n: Number of examples to generate.
        seed: Random seed.
        
    Returns:
        DataFrame of viral content features.
    """
    np.random.seed(seed)
    
    data = {
        'followers_count': np.random.lognormal(10, 1.5, n).astype(int),
        'following_count': np.random.lognormal(6, 1, n).astype(int),
        'posts_count': np.random.lognormal(6, 1.5, n).astype(int),
        'account_age_days': np.random.lognormal(7, 1, n).astype(int),
        'has_profile_picture': np.random.binomial(1, 0.95, n),
        'has_bio': np.random.binomial(1, 0.85, n),
        'content_length': np.random.lognormal(5.5, 0.7, n).astype(int),
        'has_hashtags': np.random.binomial(1, 0.7, n),
        'has_mentions': np.random.binomial(1, 0.5, n),
        'has_url': np.random.binomial(1, 0.5, n),
        'hour_of_day': np.random.choice([17, 18, 19, 20, 21], n),
        'day_of_week': np.random.choice([5, 6, 0], n, p=[0.4, 0.4, 0.2]),
        'avg_likes_history': np.random.lognormal(8, 1.5, n).astype(int),
        'avg_comments_history': np.random.lognormal(5, 1.5, n).astype(int),
    }
    
    return pd.DataFrame(data)


def generate_non_viral_examples(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate specifically non-viral content examples.
    
    Args:
        n: Number of examples to generate.
        seed: Random seed.
        
    Returns:
        DataFrame of non-viral content features.
    """
    np.random.seed(seed)
    
    data = {
        'followers_count': np.random.lognormal(5, 1.5, n).astype(int),
        'following_count': np.random.lognormal(7, 1, n).astype(int),
        'posts_count': np.random.lognormal(3, 1.5, n).astype(int),
        'account_age_days': np.random.lognormal(4, 1, n).astype(int),
        'has_profile_picture': np.random.binomial(1, 0.6, n),
        'has_bio': np.random.binomial(1, 0.4, n),
        'content_length': np.random.lognormal(4, 1, n).astype(int),
        'has_hashtags': np.random.binomial(1, 0.3, n),
        'has_mentions': np.random.binomial(1, 0.2, n),
        'has_url': np.random.binomial(1, 0.2, n),
        'hour_of_day': np.random.randint(0, 24, n),
        'day_of_week': np.random.randint(0, 7, n),
        'avg_likes_history': np.random.lognormal(3, 1, n).astype(int),
        'avg_comments_history': np.random.lognormal(1.5, 1, n).astype(int),
    }
    
    return pd.DataFrame(data)


def get_balanced_dataset(n_samples: int = 5000, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a balanced dataset with equal viral/non-viral samples.
    
    Args:
        n_samples: Total number of samples (will be split evenly).
        seed: Random seed.
        
    Returns:
        Tuple of (features DataFrame, target Series).
    """
    half = n_samples // 2
    
    viral_df = generate_viral_examples(half, seed=seed)
    viral_df['_is_viral'] = 1
    
    non_viral_df = generate_non_viral_examples(half, seed=seed + 1)
    non_viral_df['_is_viral'] = 0
    
    combined = pd.concat([viral_df, non_viral_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed)  # Shuffle
    
    return combined.drop('_is_viral', axis=1), combined['_is_viral']


if __name__ == "__main__":
    # Demo: Generate and display sample data
    print("Generating synthetic training data...")
    
    X, y = generate_synthetic_data(n_samples=1000, viral_ratio=0.3)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Viral samples: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Non-viral samples: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")
    
    print("\nFeature statistics:")
    print(X.describe())
    
    print("\nViral vs Non-Viral comparison:")
    for col in ['followers_count', 'avg_likes_history', 'posts_count']:
        print(f"\n{col}:")
        print(f"  Viral mean: {X.loc[y==1, col].mean():.1f}")
        print(f"  Non-viral mean: {X.loc[y==0, col].mean():.1f}")
