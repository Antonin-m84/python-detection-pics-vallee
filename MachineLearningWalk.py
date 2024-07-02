import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données du fichier texte en ignorant les 44 premières lignes
file_path = 'D:/Documents/Mémoire/Data John Doe/John Doe gaitway 3D locomotion_W7.txt'
data = pd.read_csv(file_path, skiprows=44, delimiter='\t')

# Extraire les colonnes temps, signal et validation
time = data.iloc[:, 0].values
signal = data.iloc[:, 19].values
validation_col = data.iloc[:, 31].values

# Définir les seuils avant et après la ligne 8500
thresholds = {
    "before_8500": {"peak": 1.935, "valley": 1.9235},
    "after_8500": {"peak": 1.9425, "valley": 1.929}
}

# Définir la taille de la fenêtre pour détecter les motifs (100 ms => 100 points à 1000 Hz)
window_size = 140

# Fonction pour extraire des caractéristiques de la fenêtre
def extract_features(signal, time, validation_col, window_size):
    features = []
    labels = []
    for i in range(len(signal) - window_size):
        window = signal[i:i + window_size]
        time_window = time[i:i + window_size]
        val_window = validation_col[i:i + window_size]

        # Déterminer les seuils en fonction de la ligne actuelle
        if i < 8500:
            peak_threshold = thresholds["before_8500"]["peak"]
            valley_threshold = thresholds["before_8500"]["valley"]
        else:
            peak_threshold = thresholds["after_8500"]["peak"]
            valley_threshold = thresholds["after_8500"]["valley"]

        # Calculer des caractéristiques simples: maximum, minimum, et l'indice de ces valeurs
        max_val = np.max(window)
        min_val = np.min(window)
        max_index = np.argmax(window)
        min_index = np.argmin(window)

        # Ajouter les caractéristiques à la liste
        features.append([max_val, min_val, max_index, min_index])

        # Définir le label en fonction de la présence du motif pic-vallée dans la fenêtre
        if (max_val >= peak_threshold and min_val <= valley_threshold
                and max_index < min_index and min_index - max_index < 140):
            # Vérifier la transition de validation_col pour confirmer la pente
            if 'DC' in val_window and 'SC' in val_window:
                labels.append(1)  # Motif trouvé
            else:
                labels.append(0)  # Motif non trouvé
        else:
            labels.append(0)  # Motif non trouvé

    return np.array(features), np.array(labels)

# Extraire les caractéristiques et les labels des données
X, y = extract_features(signal, time, validation_col, window_size)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer et entraîner un modèle de forêt aléatoire
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédire les labels pour l'ensemble de test
y_pred = model.predict(X_test)

# Calculer et afficher la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Appliquer le modèle à l'ensemble complet des données pour détecter les motifs
y_full_pred = model.predict(X)

# Extraire les temps et les valeurs des motifs détectés
patterns = []
for i in range(len(y_full_pred)):
    if y_full_pred[i] == 1:
        patterns.append((time[i], signal[i]))

# Afficher les motifs détectés
print("Motifs détectés (temps, signal):")
for pattern in patterns:
    print(pattern)
