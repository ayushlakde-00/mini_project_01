import os
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Define file paths
model_path = os.path.join(os.getcwd(), "depression_model.pkl")
vectorizer_path = os.path.join(os.getcwd(), "vectorizer.pkl")

# Check if model and vectorizer exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"❌ Vectorizer file not found: {vectorizer_path}")

# Load Model & Vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route("/", methods=["GET"])
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    # Transform input text
    text_tfidf = vectorizer.transform([user_text])

    # Predict
    prediction = model.predict(text_tfidf)
    result = "Depressed 😔" if prediction[0] == 1 else "Not Depressed 😊"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)