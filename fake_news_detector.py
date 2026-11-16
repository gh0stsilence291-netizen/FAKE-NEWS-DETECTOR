"""
Fake News / Clickbait Detector
Simple TF-IDF + Logistic Regression classifier.
Works offline after first run (model is trained on sample data).
"""

import re
import sys
import pickle
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    print("Install requirements: pip install -r requirements.txt")
    sys.exit(1)

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()


# Sample dataset (Real = 0, Fake/Clickbait = 1)
# Enough examples so the demo model works decently
SAMPLE_DATA = [
    # Real
    ("Scientists discover new planet similar to Earth in nearby star system", 0),
    ("Government announces new education policy for 2026", 0),
    ("Stock market closes higher after positive economic data", 0),
    ("Local hospital opens new cancer treatment wing", 0),
    ("Researchers develop more efficient solar panel technology", 0),
    ("City council approves new public transport project", 0),
    ("University launches free online courses for students", 0),
    ("New vaccine shows 94 percent effectiveness in clinical trials", 0),
    ("India successfully launches new satellite into orbit", 0),
    ("Heavy rainfall expected in northern states this week", 0),
    ("Tech company announces record quarterly profits", 0),
    ("New study finds link between sleep and productivity", 0),
    ("Central bank keeps interest rates unchanged this quarter", 0),
    ("National cricket team wins the series by three matches", 0),
    ("New metro line will open next month in the capital", 0),
    ("Researchers publish paper on climate change impact", 0),
    ("Company hires 500 new employees for its expansion plan", 0),
    ("Health ministry issues advisory on seasonal flu", 0),
    ("Oil prices drop slightly after global supply update", 0),
    ("School board introduces new curriculum from next year", 0),
    ("Scientists confirm water found on distant moon", 0),
    ("Parliament passes new bill on digital privacy", 0),
    ("Airport expands capacity with new terminal building", 0),
    ("Study shows regular exercise reduces heart disease risk", 0),
    ("Local farmers report good harvest this season", 0),

    # Fake / Clickbait
    ("You won't believe what this celebrity did last night", 1),
    ("Doctors hate this one weird trick that cures everything", 1),
    ("Shocking discovery drinking this every morning will change your life", 1),
    ("This simple method makes you rich in 7 days banks hate it", 1),
    ("Breaking famous actor secretly married alien from Mars", 1),
    ("Click here to see the most shocking video of the year", 1),
    ("This one food item is killing you slowly according to experts", 1),
    ("Secret government files reveal truth about the moon landing", 1),
    ("Man grows 10 inches taller using this ancient technique", 1),
    ("Scientists confirm chocolate cures all diseases overnight", 1),
    ("You are not ready for this shocking celebrity transformation", 1),
    ("Local man discovers cure for baldness in his kitchen", 1),
    ("This shocking method will make you lose 20 kg in 10 days", 1),
    ("Celebrity reveals the one product that changed her skin forever", 1),
    ("Doctors are shocked by this common kitchen item that burns fat", 1),
    ("One weird tip that banks don't want you to know about debt", 1),
    ("This ancient secret will make you look 20 years younger", 1),
    ("Breaking news celebrity caught in alien spaceship", 1),
    ("You will never believe what happened next in this video", 1),
    ("This miracle drink melts belly fat while you sleep", 1),
    ("Scientists hate him for this simple weight loss trick", 1),
    ("Shocking truth about the food you eat every day", 1),
    ("This one pill will make you rich and famous overnight", 1),
    ("Government is hiding this cure for all diseases", 1),
    ("Clickbait title that forces you to open the article now", 1),
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)


def train_model():
    texts = [clean_text(t) for t, _ in SAMPLE_DATA]
    labels = [l for _, l in SAMPLE_DATA]

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained. Sample accuracy: {acc*100:.1f}%")

    return model, vectorizer


def predict(text: str, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba) * 100
    label = "FAKE / CLICKBAIT" if pred == 1 else "REAL / LEGIT"
    return label, confidence


def main():
    print("=" * 55)
    print("   Fake News / Clickbait Detector")
    print("=" * 55)
    print()

    model, vectorizer = train_model()
    print()

    while True:
        print("-" * 40)
        text = input("Enter news headline or short article (or 'quit' to exit):\n> ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        if not text:
            continue

        label, conf = predict(text, model, vectorizer)
        print(f"\nPrediction : {label}")
        print(f"Confidence : {conf:.1f}%\n")


if __name__ == "__main__":
    main()
