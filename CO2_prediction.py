import joblib

# Charger le modèle sauvegardé
model = joblib.load('model.pkl')

# Demander à l'utilisateur d'entrer la taille du moteur
engine_size = float(input("Entrez la taille du moteur: "))

# Faire la prédiction en utilisant le modèle chargé
predicted_CO2_emissions = model.predict([[engine_size]])

# Afficher la prédiction
print(f"Les émissions de CO2 prédites pour une taille de moteur de {engine_size} sont: {predicted_CO2_emissions[0]}")
