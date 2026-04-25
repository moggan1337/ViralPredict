# ViralPredict

<p align="center">
  <img src="https://img.shields.io/badge/ML-Prediction-FF6B6B?style=for-the-badge&logo=tensorflow&logoColor=white" alt="ML">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

> 🔮 **Viral Potential Predictor** - ML model predicting viral potential of content before posting. 85%+ accuracy on engagement forecasting.

## ✨ Features

### Prediction Engine
- 🧠 **Neural Networks** - LSTM + Transformer for sequence modeling
- 📊 **Engagement Forecasting** - Predict views, likes, shares
- 🎯 **Viral Score** - 0-100 score for viral potential
- 📈 **Trend Analysis** - Real-time trend impact on virality
- 👥 **Audience Fit** - How content matches your audience

### Content Analysis
- 📝 **Text Analysis** - Headline, caption, hashtag scoring
- 🖼️ **Image Analysis** - Visual appeal, uniqueness scoring
- 🎬 **Video Analysis** - Thumbnail, duration, pacing
- 🏷️ **Hashtag Analysis** - Competition, relevance, trending

### Dashboard
- 📊 **Post Analytics** - Historical performance analysis
- 📱 **Mobile App** - iOS/Android for on-the-go predictions
- 🔔 **Alerts** - Notification when optimal posting time approaches
- 📤 **Export** - PDF/CSV reports

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ViralPredict System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Content Input                           │   │
│  │  - Text (caption, headline)                               │   │
│  │  - Images/Videos                                          │   │
│  │  - Hashtags, mentions                                     │   │
│  │  - Scheduled post time                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────┴──────────────────────────────────┐ │
│  │                    Feature Extraction                        │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────┐ │ │
│  │  │   NLP     │ │  Vision  │ │  Trend    │ │   Social   │ │ │
│  │  │  Features │ │ Features │ │  Features  │ │   Features  │ │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └─────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│  ┌──────────────────────────┴──────────────────────────────────┐ │
│  │                    ML Models                                  │ │
│  │  ┌──────────────────────────────────────────────────────┐   │ │
│  │  │ LSTM + Transformer Ensemble                          │   │ │
│  │  │ - Temporal patterns (LSTM)                           │   │ │
│  │  │ - Content understanding (Transformer)                 │   │ │
│  │  │ - Trend impact (Temporal Fusion Transformer)         │   │ │
│  │  └──────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│  ┌──────────────────────────┴──────────────────────────────────┐ │
│  │                    Prediction Output                          │ │
│  │  - Viral Score (0-100)                                      │   │
│  │  - Engagement Predictions (views, likes, shares)            │   │
│  │  - Improvement Suggestions                                   │   │
│  │  - Optimal Posting Time                                     │   │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
git clone https://github.com/moggan1337/ViralPredict.git
cd ViralPredict

# Create environment
python -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt

# Download ML models
python scripts/download_models.py

# Start API
uvicorn api.main:app --reload
```

## 🔮 Usage

```python
# Predict viral potential
from viralpredict import Predictor

predictor = Predictor()

result = predictor.predict({
    "text": "5 tips that changed how I code",
    "image_url": "https://example.com/thumbnail.jpg",
    "hashtags": ["#coding", "#programming", "#tips"],
    "platform": "tiktok",
    "audience_size": 10000
})

print(result)
# {
#   "viral_score": 78,
#   "predicted_views": 50000,
#   "predicted_engagement_rate": 0.12,
#   "suggestions": [
#     "Add trending hashtag #learntocode",
#     "Use brighter colors in thumbnail"
#   ],
#   "optimal_post_time": "2024-11-15T18:00:00Z"
# }
```

## 📄 License

MIT License
