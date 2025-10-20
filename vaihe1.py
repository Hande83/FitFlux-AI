import pandas as pd

# Ladataan datasetit, huomioiden mahdolliset pilkut desimaalierottimina, 
df1 = pd.read_csv("activities.csv", decimal=",", thousands=None)
df2 = pd.read_csv("activities_m.csv", decimal=",", thousands=None)
#yhdistetään datasetit
df = pd.concat([df1, df2], ignore_index=True)

# Tarkistetaan onko dublikaatteja
if 'Activity ID' in df.columns:
    df = df.drop_duplicates(subset=['Activity ID'], keep='first')
else:
    print("Varoitus: 'Activity ID' -sarake puuttuu. Tarkista tiedoston sisältö.")
    df = df.drop_duplicates(keep='first')  # Poistetaan duplikaatit ilman Activity ID:tä
    print(f"Yhdistetty data dublikaattien poiston jälkeen: {len(df)} riviä")

# Valitaan ominaisuudet
features = [
    "Relative Effort",
    "Average Heart Rate",
    "Average Speed",
    "Distance",
    "Elapsed Time",
    "Elevation Gain"
]
data = df[features + ["Activity Name"]].copy()  # Lisätään features ja Activity Name

# Puhdistetaan data: Käsitellään puuttuvat arvot
print("Puuttuvien arvojen määrä ennen käsittelyä:")
print(data.isna().sum())

# Varmistetaan, että kaikki sarakkeet ovat numeerisia
for column in features:
    data[column] = pd.to_numeric(data[column], errors="coerce")
    
# Poista rivit, joilta puuttuu Average Heart Rate
data = data.dropna(subset=["Average Heart Rate"])

# puuttuvat nollaksi
for column in features:
    if data[column].isna().sum() > 0:
        data[column] = data[column].fillna(0)  

# Tarkistetaan puuttuvat arvot uudelleen
print("\nPuuttuvien arvojen määrä käsittelyn jälkeen:")
print(data.isna().sum())

# Luodaan intensiteettiluokka Relative Effort:n perusteella

quartiles = data["Relative Effort"].quantile([0.25, 0.75])
q1, q3 = quartiles[0.25], quartiles[0.75]
print(f"\nRelative Effort: Q1={q1:.2f}, Q3={q3:.2f}") # pyöristetään kahteen desimaaliin

def classify_intensity(relative_effort):
    if relative_effort <= q1: # Alin 25% datasta --> "kevyt"
        return "Kevyt"
    elif q1 < relative_effort <= q3: # 50% datasta --> "keskiverto"
        return "Keskiverto"
    else:
        return "Rankka" # ylin 25 % datasta --> "rankka"

data["Intensity"] = data["Relative Effort"].apply(classify_intensity)

#Tarkistetaan, miltä data näyttää
print("\nData esikäsittelyn jälkeen (5 riviä):")
print(data.head())

# Tallennetaan esikäsitelty data desimaalipisteillä
data.to_csv("processed_activities_combi.csv", index=False, decimal=".", float_format="%.6f")
print("\nEsikäsitelty data tallennettu tiedostoon 'processed_activities_combi.csv'")

#Tulostetaan intensiteettiluokkien jakauma
print("\nIntensiteettiluokkien jakauma:")
print(data["Intensity"].value_counts())