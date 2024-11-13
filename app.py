# app.py
from flask import Flask, request, jsonify
import joblib

# Charger le modèle et le vectoriseur TF-IDF
model = joblib.load('model_spam_detection.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    label = "spam" if prediction == 1 else "ham"
    return jsonify({'prediction': label})

@app.route('/')
def home():
    return "API est run"


if __name__ == '__main__':
    app.run(debug=True)
