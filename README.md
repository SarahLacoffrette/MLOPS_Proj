
# Projet MLOps : Détection de Spam avec un Modèle Naïve Bayes

Ce projet consiste à développer un modèle de détection de spam pour classer les messages comme "spam" ou "ham" (non spam) en utilisant un modèle Naïve Bayes et la vectorisation TF-IDF. Le projet inclut un pipeline de MLOps avec surveillance des performances, réentraînement automatisé et déploiement en API.

## Table des Matières

- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Usage](#usage)
- [Fonctionnalités](#fonctionnalités)
- [Pipeline de Surveillance](#pipeline-de-surveillance)
- [Déploiement avec Docker (optionnel)](#déploiement-avec-docker-optionnel)
- [Suivi des Expérimentations avec MLflow](#suivi-des-expérimentations-avec-mlflow)

## Structure du Projet

- `train_model.py` : Script d'entraînement et de sauvegarde du modèle.
- `app.py` : API Flask pour déployer le modèle en local.
- `check_performance.py` : Script de surveillance des performances avec réentraînement conditionnel.
- `requirements.txt` : Liste des dépendances.
- `cron.log` : Fichier de log pour vérifier l’exécution du pipeline de surveillance.

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone <URL_DU_DEPOT>
   cd <NOM_DU_DEPOT>
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurer MLflow** :git init
   - Lancer MLflow en local pour suivre les expérimentations :
     ```bash
     mlflow ui
     ```
   - Accéder à MLflow via `http://127.0.0.1:5000`.

## Usage

### 1. Entraîner le Modèle
Entraîne le modèle Naïve Bayes en exécutant `train_model.py` :

```bash
python train_model.py
```

Ce script effectue :
- Le chargement et l'ingestion des données
- Le prétraitement et la vectorisation TF-IDF
- L'entraînement du modèle et l'enregistrement de la précision

### 2. Lancer l'API Flask

Démarre l'API Flask pour utiliser le modèle de détection de spam en local :

```bash
python app.py
```

L'API sera accessible à `http://127.0.0.1:5000/predict`.

Pour tester l'API :
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"message": "Congratulations! You have won a free prize."}'
```

### 3. Surveillance des Performances et Réentraînement Automatisé

Exécute `check_performance.py` manuellement pour vérifier les performances ou planifie ce script avec un `cron job` pour qu’il s’exécute périodiquement.

Exemple de configuration d’un `cron job` quotidien :
```bash
0 0 * * * /usr/bin/python3 /path/to/check_performance.py
```

## Fonctionnalités

- **Modèle Naïve Bayes pour la Détection de Spam**
- **API Flask pour le Déploiement en Local**
- **Suivi des Performances avec MLflow**
- **Réentraînement Automatisé basé sur un Seuil de Précision**
- **Pipeline de Surveillance avec Logs**

## Pipeline de Surveillance

Le script `check_performance.py` vérifie la précision du modèle et déclenche un réentraînement si la précision descend sous un seuil (90%). Les résultats sont enregistrés dans `cron.log` pour suivre les exécutions programmées.

## Déploiement avec Docker (Optionnel)

1. **Construire l’image Docker** :
   ```bash
   docker build -t spam-detector-api .
   ```

2. **Lancer le conteneur Docker** :
   ```bash
   docker run -p 5000:5000 spam-detector-api
   ```

L'API sera accessible à `http://127.0.0.1:5000/predict`.

## Suivi des Expérimentations avec MLflow

Toutes les expérimentations, y compris les performances et les paramètres du modèle, sont suivies dans MLflow pour faciliter la comparaison et la sélection de modèles optimaux.
