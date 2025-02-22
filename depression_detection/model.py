import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Sample dataset for training
data = {
    "text": [
        "I feel hopeless and alone.",
        "Life is beautiful!",
        "I'm struggling to get out of bed.",
        "Everything is great!",
        "I don't see a point in anything.",
        "I love spending time with my friends!",
        "Nothing makes me happy anymore.",
        "I'm excited about my future!",
        "I'm tired of feeling this way.",
        "Today is a fantastic day!"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Depressed, 0 = Not Depressed
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Train TF-IDF + Logistic Regression Model
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)

# Define file paths
model_path = os.path.join(os.getcwd(), "depression_model.pkl")
vectorizer_path = os.path.join(os.getcwd(), "vectorizer.pkl")

# Save Model & Vectorizer
joblib.dump(classifier, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"✅ Model trained and saved at: {model_path}")
print(f"✅ Vectorizer saved at: {vectorizer_path}")