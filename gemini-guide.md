Okay, this is a fantastic way to learn! We'll build a **Sentiment Analysis Web App**.
Users will type in a piece of text (like a movie review), and our app will predict whether the sentiment is "Positive" or "Negative".

**Why this project?**
*   **Relatable:** Everyone understands positive/negative sentiment.
*   **Sufficiently Challenging:** Involves text preprocessing, model training, and web integration.
*   **Covers Core ML Concepts:** Feature extraction (from text), model training, evaluation, and deployment (local).
*   **Practical:** Sentiment analysis is widely used.

**Tech Stack:**
*   **Python:** For ML and backend.
*   **Scikit-learn:** For ML algorithms and tools.
*   **Pandas:** For data manipulation (though we might not need it extensively for a simple version).
*   **NLTK or spaCy (Optional, for advanced text cleaning):** We'll start simple.
*   **Flask:** A lightweight Python web framework.
*   **HTML/CSS (minimal):** For the frontend.

---

**Step-by-Step Guide**

**Phase 1: The Machine Learning Model**

We'll first build and train the sentiment analysis model in a Jupyter Notebook or a Python script.

**Step 1: Project Setup & Dependencies**

1.  **Create a project directory:**
    ```bash
    mkdir sentiment_analyzer
    cd sentiment_analyzer
    ```

2.  **Create a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install necessary libraries:**
    ```bash
    pip install scikit-learn pandas numpy joblib flask
    ```
    *   `scikit-learn`: For ML.
    *   `pandas`: For handling data (especially if using a CSV).
    *   `numpy`: Scikit-learn depends on it.
    *   `joblib`: For saving/loading our trained model.
    *   `flask`: For the web app.

**Step 2: Get the Data**

For simplicity, we'll use a small, well-known dataset. The "IMDb Movie Reviews Dataset" is classic but large. Let's start with a curated, smaller sample. If you want to use the full IMDb dataset, you can download it from Kaggle.

For now, let's create a *very* simple dataset directly in our code to get started. Later, you can swap this out for a CSV.

**`train_model.py` (Create this file in your `sentiment_analyzer` directory)**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving the model and vectorizer

def train_sentiment_model():
    print("Starting model training...")

    # 1. Load Data (Simple In-Memory Dataset)
    # In a real project, you'd load this from a CSV, e.g., IMDb dataset
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

    # For a larger dataset from CSV:
    # df = pd.read_csv('your_movie_reviews.csv') # Ensure it has 'text' and 'sentiment' columns
    # print(f"Loaded dataset with {len(df)} reviews.")

    # 2. Preprocessing & Feature Extraction
    # Convert text data into numerical features using TF-IDF
    # TF-IDF: Term Frequency-Inverse Document Frequency. It weights words based on their importance in a document and across the corpus.
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000) # Limit features for simplicity
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']

    # 3. Split Data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    # Logistic Regression is a good, simple baseline model for text classification.
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save the Model and Vectorizer
    # It's CRUCIAL to save the vectorizer because you need to transform new input text
    # using the SAME vectorizer that was used during training.
    print("Saving model and vectorizer...")
    joblib.dump(model, 'sentiment_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("Model and vectorizer saved successfully!")

if __name__ == '__main__':
    train_sentiment_model()
```

**Step 3: Run the Training Script**

Open your terminal (with the virtual environment activated) in the `sentiment_analyzer` directory and run:
```bash
python train_model.py
```
You should see output indicating the training process, accuracy, and that `sentiment_model.joblib` and `tfidf_vectorizer.joblib` have been saved. The accuracy might be perfect for this tiny dataset, but on real data, it will be more nuanced.

**Key ML Concepts Covered So Far:**
*   **Dataset:** Labeled examples (text + sentiment).
*   **Feature Extraction:** `TfidfVectorizer` converts text into a matrix of numerical features. Words are features.
*   **Training/Test Split:** We train the model on one part of the data and test its performance on unseen data to get an unbiased evaluation.
*   **Model:** `LogisticRegression` learns a relationship between the TF-IDF features and the sentiment.
*   **Evaluation:** `accuracy_score` and `classification_report` tell us how well the model performs.
*   **Persistence:** `joblib` saves our trained model and vectorizer so we don't have to retrain every time.

---

**Phase 2: The Flask Web Application**

Now, let's build the web interface.

**Step 4: Create the Flask App Structure**

Your `sentiment_analyzer` directory should look like this:

```
sentiment_analyzer/
├── venv/
├── sentiment_model.joblib         # From training
├── tfidf_vectorizer.joblib      # From training
├── train_model.py                 # Our training script
└── app.py                         # NEW: Our Flask app
└── templates/                     # NEW: Directory for HTML files
    └── index.html                 # NEW: Our main HTML page
```

1.  Create the `templates` directory: `mkdir templates`

**Step 5: Create the HTML Frontend (`templates/index.html`)**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analyzer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); max-width: 600px; margin: auto; }
        h1 { color: #333; text-align: center; }
        textarea { width: 95%; padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; min-height: 100px; }
        input[type="submit"] { background-color: #5cb85c; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        input[type="submit"]:hover { background-color: #4cae4c; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; }
        .positive { background-color: #dff0d8; border: 1px solid #d6e9c6; color: #3c763d; }
        .negative { background-color: #f2dede; border: 1px solid #ebccd1; color: #a94442; }
        .neutral { background-color: #e7e7e7; border: 1px solid #ccc; color: #555; } /* Optional */
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analyzer</h1>
        <form method="POST" action="/predict">
            <textarea name="text" placeholder="Enter text here..."></textarea><br>
            <input type="submit" value="Analyze Sentiment">
        </form>

        {% if prediction %}
        <div class="result {{ prediction_class }}">
            <p><strong>Original Text:</strong> {{ original_text }}</p>
            <p><strong>Predicted Sentiment:</strong> {{ prediction }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
```

**Step 6: Create the Flask Backend (`app.py`)**

```python
from flask import Flask, request, render_template
import joblib
import numpy as np # May not be strictly needed here but good practice if dealing with model outputs directly

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and vectorizer
# Do this ONCE when the app starts
try:
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer file not found. Train the model first using train_model.py.")
    model = None
    vectorizer = None
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None


@app.route('/', methods=['GET'])
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if model is None or vectorizer is None:
        return render_template('index.html', prediction="Error: Model not loaded.", original_text="")

    if request.method == 'POST':
        text_input = request.form.get('text')
        if not text_input or not text_input.strip():
            return render_template('index.html', prediction="Error: No text provided.", original_text="")

        # Preprocess the input text using the loaded TF-IDF vectorizer
        # IMPORTANT: Use transform(), NOT fit_transform(), as the vectorizer is already fitted.
        text_vectorized = vectorizer.transform([text_input])

        # Make prediction
        prediction_raw = model.predict(text_vectorized)
        sentiment = prediction_raw[0] # Get the first (and only) prediction

        # For styling the result box
        prediction_class = "neutral" # Default
        if sentiment == 'positive':
            prediction_class = "positive"
        elif sentiment == 'negative':
            prediction_class = "negative"

        return render_template('index.html',
                               prediction=sentiment.capitalize(),
                               original_text=text_input,
                               prediction_class=prediction_class)
    return render_template('index.html') # Should not happen with POST only route for /predict

if __name__ == '__main__':
    # Make sure the model and vectorizer are loaded before running the app
    if model is None or vectorizer is None:
        print("Exiting: Model or vectorizer could not be loaded. Run train_model.py first.")
    else:
        print("Starting Flask app...")
        # For development:
        app.run(debug=True, port=5000)
        # For production (more complex, often use Gunicorn or similar):
        # app.run(host='0.0.0.0', port=80)
```

**Step 7: Run the Flask App**

1.  Make sure your virtual environment is activated.
2.  Ensure `sentiment_model.joblib` and `tfidf_vectorizer.joblib` are in the same directory as `app.py`.
3.  Run the Flask app from your terminal:
    ```bash
    python app.py
    ```
4.  Open your web browser and go to `http://127.0.0.1:5000/` (or `http://localhost:5000/`).

You should see your web page. Type some text into the textarea and click "Analyze Sentiment". The page should reload and show you the predicted sentiment.

**How the Flask App Works:**
*   **`@app.route('/')`**: Defines the main page. It renders `index.html`.
*   **`@app.route('/predict', methods=['POST'])`**: This route is called when you submit the form.
    *   It gets the text from the form (`request.form['text']`).
    *   It uses the *loaded* `tfidf_vectorizer` to convert the new text into the same numerical format the model expects (`vectorizer.transform()`). **This is a critical step.**
    *   It uses the *loaded* `model` to predict the sentiment (`model.predict()`).
    *   It then re-renders `index.html`, passing the prediction and original text back to be displayed.
*   **`render_template()`**: Flask function to render HTML files from the `templates` folder.
*   **Jinja2 Templating (`{% if prediction %}` etc. in HTML):** Allows you to embed Python-like logic and variables directly into your HTML.

---

**Phase 3: Understanding and Next Steps**

**What you've learned:**
1.  **End-to-End ML Workflow:** From data to a working web application.
2.  **Text Preprocessing:** The importance of converting text to numbers (TF-IDF).
3.  **Model Training & Evaluation:** Basic scikit-learn usage.
4.  **Model Persistence:** Saving and loading models (`joblib`).
5.  **Web Framework Basics (Flask):** Routing, handling requests, rendering templates.
6.  **Integrating ML into a Web App:** The crucial step of using the *same* preprocessing (vectorizer) for new data as was used for training.

**Challenges and Nuances:**
*   **Small Dataset:** Our current model is trained on tiny data and won't be very robust. For a real application, you'd use a much larger dataset (e.g., thousands of reviews).
*   **Vocabulary Mismatch:** If the user inputs words the `TfidfVectorizer` hasn't seen during training, those words will be ignored. `max_features` in `TfidfVectorizer` limits the vocabulary size.
*   **Simple Model:** `LogisticRegression` is good, but more complex models (like Naive Bayes for text, or even simple Neural Networks/LSTMs with libraries like TensorFlow/Keras) might perform better on larger datasets.
*   **Error Handling:** Our app has minimal error handling.
*   **Scalability:** `app.run(debug=True)` is for development. Production deployment uses tools like Gunicorn or uWSGI behind a web server like Nginx.

**Next Steps & Improvements:**
1.  **Use a Real Dataset:**
    *   Download the "IMDb Dataset of 50K Movie Reviews" from Kaggle: `https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`
    *   Modify `train_model.py` to load and parse this CSV. You'll likely have many more features when `max_features` is increased or removed. Training will take longer.
2.  **Better Text Preprocessing (in `train_model.py` and for prediction):**
    *   Lowercasing (TF-IDF does this by default).
    *   Removing punctuation.
    *   Stemming or Lemmatization (using NLTK or spaCy) to reduce words to their root form (e.g., "running" -> "run").
    ```python
    # Example using NLTK for stemming (add to train_model.py)
    # pip install nltk
    # import nltk
    # from nltk.stem import PorterStemmer
    # from nltk.tokenize import word_tokenize
    # nltk.download('punkt') # Download once
    # stemmer = PorterStemmer()
    # def stemmed_words(doc):
    #    return (stemmer.stem(w) for w in word_tokenize(doc))
    # vectorizer = TfidfVectorizer(tokenizer=stemmed_words, stop_words='english', ...)
    ```
3.  **Try Different Models:** Experiment with `MultinomialNB` (Naive Bayes) from scikit-learn, which often works well for text.
    ```python
    from sklearn.naive_bayes import MultinomialNB
    # model = MultinomialNB() # in train_model.py
    ```
4.  **More Sophisticated Evaluation:** Look at precision, recall, F1-score per class in the `classification_report`. Understand what a confusion matrix tells you.
5.  **Improve Frontend:** Make it look nicer with more CSS or even a simple JavaScript framework like Vue.js or React (though that adds complexity).
6.  **Add Confidence Scores:** Many models can output probabilities (e.g., `model.predict_proba(text_vectorized)`). You could display "Positive (90% confident)".
7.  **Deployment (Advanced):**
    *   Create a `requirements.txt`: `pip freeze > requirements.txt`
    *   Containerize with Docker.
    *   Deploy to a cloud platform (Heroku, AWS, Google Cloud).

This project should give you a solid foundation. The key is to understand each step and why it's necessary. Good luck, and have fun building!