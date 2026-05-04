# Emotion Detection — CS362 Project

https://emotion-detection-model-py.streamlit.app/

A text-based emotion detection model that classifies tweets as Joy, Sadness, or Anger
using Logistic Regression and TF-IDF vectorization.

## Dataset
- Source: Kaggle — Twitter Emotion Dataset (Parquet format)
- Place the file at: `./upload/train-00000-of-00001.parquet`
- Emotions used: Sadness (0), Joy (1), Anger (3)

## Installation
```bash
pip install -r requirements.txt
```

## Download NLTK Data (run once)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

## How to Run
1. Preprocess + Train:
```bash
python train_model.py
```
2. Evaluate + Test:
```bash
python evaluate_model.py
```

## Output Files
- `emotion_model.pkl` — trained model
- `tfidf_vectorizer.pkl` — fitted vectorizer
- `confusion_matrix.png` — evaluation chart
