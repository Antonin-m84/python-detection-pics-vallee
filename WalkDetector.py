import numpy as np
import pandas as pd

# Charger les données du fichier texte en ignorant les 44 premières lignes
file_path = 'D:/Documents/Mémoire/Data John Doe/John Doe gaitway 3D locomotion_W7.txt'
data = pd.read_csv(file_path, skiprows=44, delimiter='\t')

# Extraire les colonnes temps et signal
time = data.iloc[:, 0].values
signal = data.iloc[:, 19].values

# Appliquer une moyenne mobile (window=21, center=True)
signal_series = pd.Series(signal)
moving_average = signal_series.rolling(window=21, center=True).mean().values

# Définir les seuils avant et après la ligne 8500
thresholds = {
    "before_8500": {"peak": 1.935, "valley": 1.9235},
    "after_8500": {"peak": 1.9425, "valley": 1.929}
}

# Détection des pics et des vallées
peaks = []
valleys = []
window_size = 150

for i in range(len(moving_average) - window_size):
    window = moving_average[i:i + window_size]
    time_window = time[i:i + window_size]

    # Déterminer les seuils en fonction de la ligne actuelle
    if i < 8500:
        peak_threshold = thresholds["before_8500"]["peak"]
        valley_threshold = thresholds["before_8500"]["valley"]
    else:
        peak_threshold = thresholds["after_8500"]["peak"]
        valley_threshold = thresholds["after_8500"]["valley"]

    # Détecter les pics et les vallées dans la fenêtre
    max_val = np.max(window)
    min_val = np.min(window)
    max_index = np.argmax(window)
    min_index = np.argmin(window)

    # Vérifier les conditions de pics et vallées (pic>seuil et vallée<seuil et pic_avant_vallée et fenêtré)
    if (max_val >= peak_threshold and min_val <= valley_threshold and
            max_index < min_index and min_index - max_index < 150 and
            max_val - min_val > 0.0125):
        peaks.append((time_window[max_index], max_val))
        valleys.append((time_window[min_index], min_val))

# Créer un DataFrame pour les pics et les vallées
peaks_df = pd.DataFrame(peaks, columns=['Time', 'Value'])
valleys_df = pd.DataFrame(valleys, columns=['Time', 'Value'])

# Calculer la différence de temps entre les pics/vallées consécutifs
peaks_df['Time_Diff'] = peaks_df['Time'].diff()
valleys_df['Time_Diff'] = valleys_df['Time'].diff()

# Filtrer les DataFrames pour ne conserver que les lignes où la différence de temps est supérieure à 0.3
peaks_df = peaks_df[peaks_df['Time_Diff'] > 0.3]
valleys_df = valleys_df[valleys_df['Time_Diff'] > 0.3]

# Ajouter le nombre de pics et vallées trouvés
peaks_count = len(peaks_df)
valleys_count = len(valleys_df)

# Créer un fichier Excel et ajouter les données
with pd.ExcelWriter('detected_patterns_W7.xlsx') as writer:
    peaks_df.to_excel(writer, sheet_name='Peaks', index=False)
    valleys_df.to_excel(writer, sheet_name='Valleys', index=False)

print(f'Nombre de pics : {peaks_count}')
print(f'Nombre de vallées : {valleys_count}')
print("Detection complète, résultats sauvegardés dans detected_patterns.xlsx")
