import joblib
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def check_model_performance():
    with open("cron/cron.log", "a") as log_file:
        log_file.write(f"\nRunning check at {datetime.now()}\n")

    # Charger le modèle et les données de test
    model = joblib.load('model_spam_detection.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    test_data = pd.read_csv('spam_ham_dataset.csv')
    X_test = vectorizer.transform(test_data['text'])

    # Encoder les labels de test
    label_encoder = LabelEncoder()
    y_test = label_encoder.fit_transform(test_data['label'])

    # Calculer la précision
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Current accuracy: {accuracy}")

    with open("cron/cron.log", "a") as log_file:
        log_file.write(f"Accuracy: {accuracy}\n")

    # Si la précision est inférieure au seuil, réentraîner le modèle
    threshold = 0.90
    if accuracy < threshold:
        log_file.write("Accuracy below threshold, retraining model...\n")
        retrain_model()

def retrain_model():
    # Réentraîner le modèle (le code d'entraînement de train_model.py)
    pass  # Inclure ici le code d'entraînement

# Exécuter la vérification
check_model_performance()
