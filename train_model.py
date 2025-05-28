import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_sentiment_model():
    print("starting model training...")

    df = pd.read_csv('IMDB-dataset.csv')

    print("vectorizing text data...")
    vectorizer = TfidfVectorizer(stop_words='english')
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
    