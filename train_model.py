import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_sentiment_model():
    print("starting model training...")

    data = {
        'text': [
            "I love this movie, it's fantastic!",
            "This was an amazing experience.",
            "Absolutely brilliant, would watch again.",
            "The best film I've seen all year.",
            "Such a heartwarming story.",
            "I hated this film, it was terrible.",
            "A complete waste of time.",
            "The acting was awful and the plot was boring.",
            "I would not recommend this to anyone.",
            "This is the worst movie ever."
        ],
        'sentiment': [
            'positive', 'positive', 'positive', 'positive', 'positive',
            'negative', 'negative', 'negative', 'negative', 'negative'
        ]
    }
    df = pd.DataFrame(data)

    print("vectorizing text data...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['text']) # X is now a matrix where each row represents a review, and each column is a word feature
    y = df['sentiment']

    print("splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("training logistic regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"model accuracy: {accuracy:.4f}")
    print("calssification report:")
    print(classification_report(y_test, y_pred))

    print("saving model and vectorizer...")
    joblib.dump(model, 'sentiment_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("model and vectorizer saved successfully")

if __name__ == '__main__':
    train_sentiment_model()
    