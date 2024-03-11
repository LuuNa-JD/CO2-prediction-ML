# Prédiction des émissions de CO2 des véhicules

Ce dépôt GitHub contient un projet de machine learning qui vise à prédire les émissions de CO2 des véhicules en fonction de la taille de leur moteur (volume cylindrique). Le modèle a été entraîné pour établir une corrélation entre la taille du moteur d'un véhicule et ses émissions de CO2, fournissant ainsi une estimation utile des émissions basée sur une caractéristique facilement mesurable du véhicule.

## Fonctionnement

Le script principal utilise un modèle de machine learning sauvegardé pour prédire les émissions de CO2 d'un véhicule basées sur la taille de son moteur. Voici un guide rapide pour utiliser ce projet :

### Prérequis

- Python 3.x
- joblib
- scikit-learn

Assurez-vous que Python et les dépendances nécessaires sont installés. Vous pouvez installer les dépendances requises en utilisant pip :

```bash
pip install joblib scikit-learn
```

## Usage

Pour prédire les émissions de CO2 d'un véhicule, suivez ces étapes :

Ouvrez votre terminal ou invite de commande.
Naviguez jusqu'au dossier contenant le projet.
Exécutez le script CO2_prediction.py avec Python :

```bash
python CO2_prediction.py
```

Entrez la taille du moteur du véhicule lorsque vous y êtes invité, et le modèle vous fournira une estimation des émissions de CO2.

### Exemple d'utilisation

```bash
Entrez la taille du moteur: 2.5
Les émissions de CO2 prédites pour une taille de moteur de 2.5 sont: 200 g/km
```
## Contribuer
Les contributions à ce projet sont les bienvenues. Si vous avez une suggestion pour améliorer ce modèle ou si vous souhaitez ajouter de nouvelles fonctionnalités, n'hésitez pas à créer une issue ou une pull request.


