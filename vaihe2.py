import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Ladataan esikäsitelty data 
data = pd.read_csv("processed_activities_combi.csv", decimal=".")

# Tarkista datan koko ja ominaisuudet
print(f"Datan koko: {data.shape}")
print("\nOminaisuuksien kuvaus:")
print(data["Relative Effort"].describe())

# Määritellään ominaisuudet. 
features = ["Average Heart Rate", "Average Speed", 
            "Distance", 
            "Elapsed Time", "Elevation Gain"]
for column in features:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Erotetaan ominaisuudet (X) ja kohdemuuttuja (y)
X = data[features]
y = data["Intensity"]

# Jaetaan data koulutus- ja testisetiksi. data 80% koulutus ja 20% testi dataan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Koulutetaan päätöspuu-malli
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Ennustetaan testidatalla ja arvioidaan tarkkuus
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nMallin tarkkuus testidatalla: {accuracy:.2f}")

# Tulostetaan confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Kevyt", "Keskiverto", "Rankka"])
print("\nConfusion Matrix:")
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Kevyt", "Keskiverto", "Rankka"], 
            yticklabels=["Kevyt", "Keskiverto", "Rankka"])
plt.xlabel("Ennustettu")
plt.ylabel("Todellinen")
plt.title("Confusion Matrix")
plt.show()

# Ominaisuuksien merkitys
importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nOminaisuuksien merkitys:")
print(importance_df)

# Tallennetaan malli
joblib.dump(model, "model.pkl")
print("\nMalli tallennettu tiedostoon 'model.pkl'")

# Esimerkki: Ennustetaan yksi harjoitus
example_row = X_test.iloc[0:1]
prediction = model.predict(example_row)
prediction_proba = model.predict_proba(example_row)  # Lisätty todennäköisyydet

# Haetaan konteksti (Activity Name)
actual_activity = data.loc[example_row.index, 'Activity Name'].iloc[0]
actual_class = y_test.iloc[0]
# Muutetaan aika minuuteiksi
example_dict = example_row.iloc[0].to_dict()
example_dict['Elapsed Time (min)'] = round(example_dict['Elapsed Time'] / 60,2)  # Muunnos minuuteiksi
del example_dict['Elapsed Time'] # Poistetaan sekunnit

print(f"\nEsimerkki ennuste:")
print(f"Syötearvot: {example_dict}")
print(f"Harjoitustyyppi: {actual_activity}")
print(f"Todellinen luokka: {actual_class}")
print(f"Ennustettu luokka: {prediction[0]}")
probs = {k: f"{v:.2f}" for k, v in dict(zip(model.classes_, prediction_proba[0])).items()}
print(f"Todennäköisyydet: {probs}")