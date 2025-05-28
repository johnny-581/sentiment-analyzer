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