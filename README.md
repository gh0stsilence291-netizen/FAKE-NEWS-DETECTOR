# Fake News / Clickbait Detector

Simple NLP project that classifies news headlines/articles as Real or Fake/Clickbait using TF-IDF + Logistic Regression.

Trained on a small sample dataset + works with custom text.

## Features
- Classify any news headline or short article
- Shows confidence score
- Lightweight, no GPU needed
- Easy to retrain with more data

## How to Run (Google Colab)

1. Open Google Colab
2. Copy code from ``fake_news_detector.py` + **`Fake_News_Detector.ipynb`**`
3. Run cells
4. Enter any news headline → get Real / Fake prediction

## Installation (Local)

```bash
pip install -r requirements.txt
python `fake_news_detector.py` + **`Fake_News_Detector.ipynb`**
```

## Tech Stack
- Python
- scikit-learn (TF-IDF + LogisticRegression)
- NLTK

## Project Structure
```
02-fake-news-detector/
├── `fake_news_detector.py` + **`Fake_News_Detector.ipynb`**
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Why this project?
Shows classic supervised NLP pipeline. Great talking point for AI Intern interviews.

## Author
BUILD BY MUHAMMAD ALI HASSAN.
