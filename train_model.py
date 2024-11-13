# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Charger les données
data = pd.read_csv('spam_ham_dataset.csv')
data = data[['label', 'text']]
data.columns = ['label', 'text']

# Encoder les labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Transformer les messages en vecteurs TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Démarrer une nouvelle expérimentation dans MLflow
mlflow.start_run(run_name="Naive_Bayes_Spam_Classifier")

# Entraîner le modèle
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Enregistrer les paramètres et métriques dans MLflow
mlflow.log_param("Model", "MultinomialNB")
mlflow.log_param("Vectorizer", "TF-IDF")
mlflow.log_metric("accuracy", accuracy)

# Enregistrer le modèle et le vectoriseur TF-IDF
mlflow.sklearn.log_model(model, "model")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Terminer l’expérimentation MLflow
mlflow.end_run()
