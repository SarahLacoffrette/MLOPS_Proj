# predict.py
import joblib

# Charger le modèle et le vectoriseur TF-IDF
model = joblib.load('model_spam_detection.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_message(message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    label = "spam" if prediction == 1 else "ham"
    return label

# Test
print(predict_message("Congratulations! You have won a free prize."))
