# ViralPredict

ML model for predicting social media content viral potential.

## Overview

ViralPredict uses machine learning to predict whether social media content will go viral based on engagement metrics, user profile data, and content features.

## Features

- **RandomForest Classifier** for robust binary classification
- **Feature Engineering** including engagement rates, ratio metrics, and temporal patterns
- **Confidence Scoring** with probability-based predictions
- **Model Persistence** using joblib for saving/loading trained models
- **Comprehensive Tests** with pytest

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.model import ViralPredictor
from src.data import generate_synthetic_data
from src.predict import predict_virality, get_sample_features

# Generate training data
X, y = generate_synthetic_data(n_samples=5000, viral_ratio=0.3)

# Create and train model
model = ViralPredictor()
metrics = model.train(X.values, y.values)

print(f"Model accuracy: {metrics['accuracy']:.2%}")
print(f"ROC AUC: {metrics['roc_auc']:.2f}")

# Make predictions
features = get_sample_features(
    followers_count=50000,
    following_count=200,
    posts_count=1000,
    content_length=300,
    has_hashtags=True,
    hour_of_day=18,
    day_of_week=5,
    avg_likes_history=5000
)

result = predict_virality(features, model=model)
print(f"Will go viral: {result['will_go_viral']}")
print(f"Confidence: {result['confidence_score']:.1%}")
```

## Features

The model uses 18 features:

| Feature | Description |
|---------|-------------|
| `followers_count` | Number of followers |
| `following_count` | Number of accounts followed |
| `posts_count` | Total posts |
| `has_profile_picture` | Profile picture present |
| `has_bio` | Bio present |
| `account_age_days` | Account age |
| `content_length` | Length of content |
| `has_hashtags` | Contains hashtags |
| `has_mentions` | Contains mentions |
| `has_url` | Contains URL |
| `hour_of_day` | Hour posted (0-23) |
| `day_of_week` | Day of week (0-6) |
| `is_weekend` | Posted on weekend |
| `engagement_rate_history` | Historical engagement rate |
| `avg_likes_history` | Average likes |
| `avg_comments_history` | Average comments |
| `follower_to_following_ratio` | Follower/following ratio |
| `posts_per_day` | Posting frequency |

## API

### ViralPredictor

```python
from src.model import ViralPredictor

# Initialize
model = ViralPredictor()

# Train
metrics = model.train(X, y, test_size=0.2, cv_folds=5)

# Predict single
result = model.predict_single(features)

# Predict batch
predictions, probabilities = model.predict(X)

# Save/Load
model.save_model('model.joblib')
model = ViralPredictor(model_path='model.joblib')
```

### predict_virality()

```python
from src.predict import predict_virality

result = predict_virality(features, model=model)
# Returns: {'will_go_viral': bool, 'confidence_score': float, ...}
```

## Testing

```bash
pytest tests/ -v
```

## Data Generation

```python
from src.data import generate_synthetic_data, get_balanced_dataset

# Random viral ratio
X, y = generate_synthetic_data(n_samples=10000, viral_ratio=0.3)

# Balanced dataset
X, y = get_balanced_dataset(n_samples=5000)
```

## Model Performance

On synthetic data with clear viral patterns:
- Accuracy: >85%
- ROC AUC: >0.90
- Cross-validation ROC AUC: Consistent across folds

## Files

```
ViralPredict/
├── src/
│   ├── __init__.py
│   ├── model.py      # ViralPredictor class
│   ├── predict.py    # Prediction utilities
│   └── data.py       # Data generation
├── tests/
│   ├── __init__.py
│   └── test_model.py  # Test suite
├── requirements.txt
└── README.md
```

## License

MIT License
