import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chemins des fichiers
file_path = r'D:\\Documents\\Mémoire\\Data John Doe\\John Doe gaitway 3D locomotion_R10.txt'
output_file = r'D:\\Documents\\Mémoire\\Data John Doe\\Auto Python\\ResultPython_JD_R10.txt'

# Seuils pour la détection des pics et vallées
seuil_p = 2.795
seuil_v = 2.755

# Lire le fichier en ignorant les 43 premières lignes
data = pd.read_csv(file_path, delimiter='\t', header=None, skiprows=44)

# Extraire les colonnes nécessaires
time = data[0]
raw_speed = data[9]
speed = data[19]
contact_mode = data[31]

# Calculer la moyenne mobile pour raw_speed
moving_average_raw_speed = raw_speed.rolling(window=21, center=True).mean()

# Initialiser les listes pour stocker les vallées, les pics et les transitions de contact mode
vallees = []
pics = []
averages_vals = []

# Détecter les vallées et les pics pour raw_speed
previous_type = None

for i in range(10, len(moving_average_raw_speed) - 10):
    if pd.notna(moving_average_raw_speed[i]):  # Vérifier que la valeur n'est pas NaN
        window = moving_average_raw_speed[i - 10:i + 11]
        min_value = window.min()
        max_value = window.max()

        is_vallee = moving_average_raw_speed[i] == min_value and moving_average_raw_speed[i] < seuil_v
        is_pic = moving_average_raw_speed[i] == max_value and moving_average_raw_speed[i] > seuil_p

        averages_vals.append((time[i], moving_average_raw_speed[i], ''))

        # Vérifier les pics
        if is_pic and previous_type != "pic":
            pics.append((time[i], moving_average_raw_speed[i], 'Pic'))
            previous_type = "pic"

        # Vérifier les vallées
        elif is_vallee and previous_type != "val":
            vallees.append((time[i], moving_average_raw_speed[i], 'Vallée'))
            previous_type = "val"

print("Dernière détection de vallée:", is_vallee)
print("Dernière détection de pic:", is_pic)

# Visualisation des données
plt.figure(figsize=(10, 6))
pics_df = pd.DataFrame(pics)
vallees_df = pd.DataFrame(vallees)
averages_vals_df = pd.DataFrame(averages_vals)

plt.plot(averages_vals_df[0], averages_vals_df[1], label='Moyenne mobile de Raw Speed', color='orange')

for i in range(len(vallees_df)):
    plt.axvline(x=vallees_df[0][i], color='green', label='Vallées' if i == 0 else "")

for i in range(len(pics_df)):
    plt.axvline(x=pics_df[0][i], color='blue', label='Pics' if i == 0 else "")

plt.xlabel('Temps')
plt.ylabel('Raw Speed')
plt.title('Raw Speed et Moyenne Mobile avec Pics et Vallées')
plt.legend()
plt.grid(True)
plt.show()
