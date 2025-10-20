# 🧠 FitFlux AI – Harjoitusten intensiteettiluokittelija / Smart Workout Intensity Classifier

FitFlux AI on Python-pohjainen projekti, joka hyödyntää Stravasta ladattua urheiludataa harjoitusten intensiteetin luokitteluun.  
Projektissa käsitellään käyttäjän omaa harjoitusdataa, esikäsitellään se ja opetetaan päätöspuuhun perustuva koneoppimismalli, joka arvioi harjoituksen rasittavuutta eri muuttujien perusteella.

FitFlux AI is a Python-based machine learning project that analyzes Strava workout data and classifies exercise intensity levels.  
The goal is to transform raw fitness data into meaningful insights using data science and AI.

---

## 📊 Projektin rakenne / Project Structure

Projekti koostuu kahdesta päävaiheesta / The project consists of two main phases:

### 1️⃣ `vaihe1.py` – Datan esikäsittely / Data Preprocessing
This phase handles all raw data from Strava exports, merges them, and prepares a clean dataset for model training.

- Lataa kaksi Strava CSV-tiedostoa (`activities.csv`, `activities_m.csv`)  
- Yhdistää datat ja poistaa duplikaatit  
- Käsittelee puuttuvat arvot ja varmistaa numeerisen muodon  
- Laskee *Relative Effort* -muuttujan kvartiilit ja luokittelee harjoitukset kolmeen intensiteettiluokkaan:  
  - 🟢 **Kevyt / Light** – alin 25 %  
  - 🟡 **Keskiverto / Moderate** – keskimmäiset 50 %  
  - 🔴 **Rankka / Hard** – ylin 25 %  
- Tallentaa esikäsitellyn datan tiedostoon `processed_activities_combi.csv`

### 2️⃣ `vaihe2.py` – Mallin koulutus ja arviointi / Model Training & Evaluation
This phase uses the cleaned dataset to train and test a Decision Tree model for workout intensity classification.

- Lataa esikäsitellyn datan  
- Jakaa datan 80/20 koulutus- ja testijoukkoon  
- Kouluttaa **DecisionTreeClassifier**-mallin (Scikit-learn)  
- Arvioi mallin tarkkuuden ja visualisoi *Confusion Matrixin* (Seaborn & Matplotlib)  
- Näyttää ominaisuuksien tärkeysjärjestyksen  
- Tallentaa mallin tiedostoon `model.pkl`  
- Tulostaa esimerkkiharjoituksen ennusteen ja todennäköisyydet  

---

## 💡 In English — Short Summary

FitFlux AI demonstrates a complete data pipeline:
1. **Data Processing:** Cleaning and merging fitness data exported from Strava  
2. **Feature Engineering:** Using metrics like Relative Effort, Heart Rate, Distance, and Elevation Gain  
3. **Model Training:** Building a decision tree classifier to categorize workouts  
4. **Evaluation:** Visualizing model accuracy, feature importance, and confusion matrix  
5. **Prediction:** Generating live examples and probabilities for new workouts