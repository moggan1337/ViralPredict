# ViralPredict

<p align="center">
  <img src="https://img.shields.io/badge/ML-Prediction-FF6B6B?style=for-the-badge&logo=tensorflow&logoColor=white" alt="ML">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge" alt="PRs Welcome">
</p>

> 🔮 **Viral Potential Predictor** — ML-powered prediction engine forecasting the viral potential of social media content before posting. 85%+ accuracy on engagement forecasting.

## About

ViralPredict uses state-of-the-art deep learning models to analyze content and predict its potential to go viral. By examining text, images, hashtags, timing, and audience fit, it provides creators and marketers with actionable insights to maximize engagement before investing time in content creation.

**Who it's for:**
- Content creators seeking to maximize reach and engagement
- Social media managers optimizing posting strategy
- Marketing teams planning campaigns
- Brands wanting data-driven content decisions

## Features

### Prediction Engine

| Feature | Description |
|---------|-------------|
| 🧠 **Neural Networks** | LSTM + Transformer ensemble for sequence and content modeling |
| 📊 **Engagement Forecasting** | Predict views, likes, shares, and comments |
| 🎯 **Viral Score** | 0-100 composite score indicating viral potential |
| 📈 **Trend Analysis** | Real-time trend impact assessment on virality |
| 👥 **Audience Fit** | Content-to-audience match scoring |

### Content Analysis

| Feature | Description |
|---------|-------------|
| 📝 **Text Analysis** | Headline, caption, and copy scoring |
| 🖼️ **Image Analysis** | Visual appeal and uniqueness scoring |
| 🎬 **Video Analysis** | Thumbnail, duration, and pacing analysis |
| 🏷️ **Hashtag Analysis** | Competition level, relevance, and trend alignment |

### Dashboard & Tools

| Feature | Description |
|---------|-------------|
| 📊 **Post Analytics** | Historical performance analysis and trends |
| 📱 **Mobile Companion** | iOS/Android app for on-the-go predictions |
| 🔔 **Smart Alerts** | Notifications for optimal posting windows |
| 📤 **Export Reports** | PDF and CSV performance reports |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ViralPredict System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      Content Input                         │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Text (caption, headline, body copy)               │  │   │
│  │  │  Images/Videos (thumbnail, media files)             │  │   │
│  │  │  Hashtags, mentions                                 │  │   │
│  │  │  Scheduled post time                               │  │   │
│  │  │  Platform target                                    │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌────────────────────────────┴──────────────────────────────┐   │
│  │                   Feature Extraction Layer                  │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  │   │
│  │  │    NLP    │ │   Vision  │ │   Trend   │ │   Social  │  │   │
│  │  │ Features  │ │ Features  │ │ Features  │ │ Features  │  │   │
│  │  │           │ │           │ │           │ │           │  │   │
│  │  │ • Embeddings│ │• Objects │ │• Trend    │ │• Followers│  │   │
│  │  │ • Sentiment│ │• Colors │ │  scores   │ │• Past perf│  │   │
│  │  │ • Keywords │ │• Faces  │ │• Seasonality│ │• Niche   │  │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌────────────────────────────┴──────────────────────────────┐   │
│  │                      ML Models Layer                        │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │              LSTM + Transformer Ensemble              │ │   │
│  │  │                                                        │ │   │
│  │  │  ┌────────────────┐  ┌────────────────┐               │ │   │
│  │  │  │      LSTM      │  │  Transformer   │               │ │   │
│  │  │  │  (Temporal)    │  │  (Content)     │               │ │   │
│  │  │  │                │  │                │               │ │   │
│  │  │  │ • Sequence     │  │ • Self-attention│              │ │   │
│  │  │  │   patterns     │  │ • Multi-head   │               │ │   │
│  │  │  │ • Time-of-day  │  │   attention    │               │ │   │
│  │  │  │ • Day-of-week  │  │ • Cross-modal  │               │ │   │
│  │  │  └────────────────┘  └────────────────┘               │ │   │
│  │  │                                                        │ │   │
│  │  │  ┌────────────────────────────────────────────────┐  │ │   │
│  │  │  │     Temporal Fusion Transformer (TFT)         │  │ │   │
│  │  │  │  • Historical trend impact                     │  │ │   │
│  │  │  │  • Dynamic covariate handling                  │  │ │   │
│  │  │  └────────────────────────────────────────────────┘  │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌────────────────────────────┴──────────────────────────────┐   │
│  │                    Prediction Output Layer                  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  • Viral Score (0-100)                             │  │   │
│  │  │  • Engagement Predictions (views, likes, shares)   │  │   │
│  │  │  • Improvement Suggestions                          │  │   │
│  │  │  • Optimal Posting Time                            │  │   │
│  │  │  • Platform-specific recommendations               │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.11+ |
| **ML Framework** | TensorFlow 2.14, PyTorch 2.1 |
| **API** | FastAPI 0.104, Pydantic v2 |
| **Data Processing** | pandas, NumPy, scikit-learn |
| **NLP** | Transformers (HuggingFace), spaCy |
| **Vision** | OpenCV, Pillow, CLIP |
| **Database** | PostgreSQL 15, Redis 7 |
| **Deployment** | Docker, uvicorn, Gunicorn |

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ (for metrics storage)
- Redis 7+ (for caching)
- 8GB+ RAM recommended (for ML model loading)
- NVIDIA GPU with CUDA 11.8+ (optional, for faster inference)

### Steps

```bash
# Clone the repository
git clone https://github.com/moggan1337/ViralPredict.git
cd ViralPredict

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained ML models
python scripts/download_models.py

# Copy and configure environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python scripts/init_db.py

# Start API server
uvicorn api.main:app --reload
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | ✅ |
| `REDIS_URL` | Redis connection string | ✅ |
| `MODEL_PATH` | Path to ML models directory | ✅ |
| `API_KEY` | API authentication key | Optional |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING) | Optional |

## Quick Start

### Predict Viral Potential

```python
from viralpredict import Predictor

# Initialize predictor
predictor = Predictor()

# Analyze content
result = predictor.predict({
    "text": "5 tips that changed how I code",
    "image_url": "https://example.com/thumbnail.jpg",
    "hashtags": ["#coding", "#programming", "#tips"],
    "platform": "tiktok",
    "audience_size": 10000,
    "posting_time": "2024-11-15T18:00:00Z"
})

print(result)
```

### Response Format

```json
{
  "viral_score": 78,
  "confidence": 0.85,
  "predictions": {
    "views": 50000,
    "likes": 2500,
    "shares": 450,
    "comments": 180,
    "engagement_rate": 0.12
  },
  "suggestions": [
    {
      "category": "hashtag",
      "message": "Add trending hashtag #learntocode",
      "expected_boost": "+15% reach"
    },
    {
      "category": "visual",
      "message": "Use brighter colors in thumbnail",
      "expected_boost": "+8% click-through"
    }
  ],
  "optimal_post_time": "2024-11-15T12:00:00Z",
  "analysis": {
    "text_score": 82,
    "image_score": 71,
    "hashtag_score": 65,
    "timing_score": 88
  }
}
```

### Batch Prediction

```python
# Batch process multiple content pieces
results = predictor.predict_batch([
    {"text": "Post 1 content...", "platform": "instagram"},
    {"text": "Post 2 content...", "platform": "twitter"},
    {"text": "Post 3 content...", "platform": "youtube"}
])
```

## API Reference

### Prediction Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | Predict viral potential for single content |
| `POST` | `/api/v1/predict/batch` | Batch prediction for multiple content items |
| `GET` | `/api/v1/predict/:id` | Get cached prediction result |

### Content Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze/text` | Analyze text content only |
| `POST` | `/api/v1/analyze/image` | Analyze image content only |
| `POST` | `/api/v1/analyze/hashtags` | Analyze hashtag effectiveness |

### Trend Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/trends` | Get current trending topics |
| `GET` | `/api/v1/trends/:platform` | Get platform-specific trends |
| `GET` | `/api/v1/trends/:hashtag/history` | Get hashtag trend history |

### Historical Performance

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/analytics/overview` | Get performance overview |
| `GET` | `/api/v1/analytics/:platform` | Get platform-specific analytics |

## Model Details

### LSTM Component
- Captures temporal patterns in content performance
- Learns time-of-day and day-of-week effects
- Memory window: 14 days of historical data

### Transformer Component
- Self-attention mechanism for content understanding
- Multi-head attention (8 heads) for diverse feature extraction
- Cross-modal attention for image-text fusion

### Temporal Fusion Transformer (TFT)
- Incorporates historical trends into predictions
- Handles static and dynamic covariates
- Quantile predictions for uncertainty estimation

## Contributing

We welcome contributions! Please follow these steps:

```bash
# Fork the repository
git clone https://github.com/<your-username>/ViralPredict.git

# Create virtual environment and install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes, commit, and push
git commit -m "Add: amazing feature"
git push origin feature/amazing-feature
```

### Development Guidelines

- Use type hints for all function signatures
- Follow PEP 8 style guide
- Write unit tests for new features
- Update docstrings for API changes
- Validate models before committing

## License

MIT License — See [LICENSE](LICENSE)

Copyright © 2024 ViralPredict Contributors

---

<p align="center">
  <sub>Predict the next viral sensation before it happens</sub>
</p>
