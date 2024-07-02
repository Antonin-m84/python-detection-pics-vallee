import pandas as pd
import numpy as np
import os
from scipy.stats import linregress
import json

def read_and_process_file(file_path, seuil_v=4.35, seuil_p=4.55):
    if not os.path.exists(file_path):
        print(f"Le fichier spécifié n'existe pas : {file_path}")
        return None, None, None

    # Lire le fichier en ignorant les 43 premières lignes
    data = pd.read_csv(file_path, delimiter='\t', header=None, skiprows=44)

    # Extraire les colonnes nécessaires
    time = data[0]
    speed = data[19]
    contact_mode = data[31]

    # Initialiser les listes pour stocker les vallées, les pics et les transitions de contact mode
    vallees = []
    pics = []
    transitions = []

    # Previous value, should be "PIC", or "VAL"
    was_previous = None

    # Calculer la moyenne mobile
    moving_average = speed.rolling(window=21, center=True).mean()

    previous_type = None

    # Détecter les vallées et les pics
    for i in range(10, len(moving_average) - 10):
        if pd.notna(moving_average[i]):  # Vérifier que la valeur n'est pas NaN
            window = moving_average[i - 10:i + 11]
            min_value = window.min()
            max_value = window.max()

            is_vallee = moving_average[i] == min_value and moving_average[i] < seuil_v
            is_pic = moving_average[i] == max_value and moving_average[i] > seuil_p

            # Vérifier les pics
            if is_pic and previous_type != "pic":
                pics.append((time[i], moving_average[i], 'Pic'))
                previous_type = "pic"

            # Vérifier les vallées
            elif is_vallee and previous_type != "val":
                vallees.append((time[i], moving_average[i], 'Vallée'))
                previous_type = "val"


    # Détecter les transitions de contact_mode
    previous_mode = None
    for i in range(len(contact_mode)):
        if pd.notna(contact_mode[i]):
            current_mode = contact_mode[i]
            if (previous_mode is None or previous_mode != 'SC') and current_mode == 'SC':
                transitions.append((time[i], 'Transition'))
            previous_mode = current_mode

    # Créer des DataFrames pour les vallées, les pics et les transitions
    df_vallees = pd.DataFrame(vallees, columns=['Temps (ms)', 'Vitesse (m/s)', 'Type'])
    df_pics = pd.DataFrame(pics, columns=['Temps (ms)', 'Vitesse (m/s)', 'Type'])
    df_transitions = pd.DataFrame(transitions, columns=['Temps (ms)', 'Type'])

    # Calculer la différence de temps entre chaque valeur x+1 et x
    df_vallees['Différence de Temps (ms)'] = df_vallees['Temps (ms)'].diff().shift(-1).abs()
    df_pics['Différence de Temps (ms)'] = df_pics['Temps (ms)'].diff().shift(-1).abs()
    df_transitions['Différence de Temps (ms)'] = df_transitions['Temps (ms)'].diff().shift(-1).abs()

    # Arrondir les valeurs de vitesse et de différence de temps
    df_vallees['Vitesse (m/s)'] = df_vallees['Vitesse (m/s)'].round(5)
    df_pics['Vitesse (m/s)'] = df_pics['Vitesse (m/s)'].round(5)
    df_vallees['Différence de Temps (ms)'] = df_vallees['Différence de Temps (ms)'].round(4)
    df_pics['Différence de Temps (ms)'] = df_pics['Différence de Temps (ms)'].round(4)
    df_transitions['Différence de Temps (ms)'] = df_transitions['Différence de Temps (ms)'].round(4)

    # Ajouter les codes d'erreur pour les différences de temps
    df_vallees['Erreur'] = df_vallees['Différence de Temps (ms)'].apply(lambda x: 'Erreur' if x > 400 else ('Avertissement' if x < 0.250 else ''))
    df_pics['Erreur'] = df_pics['Différence de Temps (ms)'].apply(lambda x: 'Erreur' if x > 400 else ('Avertissement' if x < 0.250 else ''))
    df_transitions['Erreur'] = df_transitions['Différence de Temps (ms)'].apply(lambda x: 'Erreur' if x > 400 else ('Avertissement' if x < 0.250 else ''))

    # Retourner les résultats
    return df_vallees, df_pics, df_transitions

def write_results_to_file(vallees, pics, transitions, output_file):
    with open(output_file, 'w') as f:
        f.write("Vallées détectées:\t")
        if not vallees.empty:
            f.write(f"({len(vallees)})\n")
            vallees.to_csv(f, index=False, sep='\t', lineterminator="\n")
            # Vérifier les différences de temps pour les vallées
            for index, row in vallees.iterrows():
                if row['Différence de Temps (ms)'] > 400 or row['Différence de Temps (ms)'] < 0.250:
                    f.write(f"Erreur: Différence de temps {row['Différence de Temps (ms)']} ms à l'index {index} pour les vallées\n")
        else:
            f.write("Aucune vallée détectée.\n")

        f.write("\nPics détectés:\n")
        if not pics.empty:
            pics.to_csv(f, index=False, sep='\t', lineterminator="\n")
            # Vérifier les différences de temps pour les pics
            for index, row in pics.iterrows():
                if row['Différence de Temps (ms)'] > 400 or row['Différence de Temps (ms)'] < 0.250:
                    f.write(f"Erreur: Différence de temps {row['Différence de Temps (ms)']} ms à l'index {index} pour les pics\n")
        else:
            f.write("Aucun pic détecté.\n")

        f.write("\nTransitions de contact_mode détectées:\t")
        if not transitions.empty:
            f.write(f"({len(vallees)})\n")
            transitions.to_csv(f, index=False, sep='\t', lineterminator="\n")
            # Vérifier les différences de temps pour les transitions
            for index, row in transitions.iterrows():
                if row['Différence de Temps (ms)'] > 400 or row['Différence de Temps (ms)'] < 0.250:
                    f.write(f"Erreur: Différence de temps {row['Différence de Temps (ms)']} ms à l'index {index} pour les transitions\n")
        else:
            f.write("Aucune transition détectée.\n")

def compare_differences(vallees, pics, transitions):
    if not vallees.empty and not pics.empty and not transitions.empty:
        vallees_diff = vallees['Différence de Temps (ms)'].dropna().values
        pics_diff = pics['Différence de Temps (ms)'].dropna().values
        transitions_diff = transitions['Différence de Temps (ms)'].dropna().values

        # Calculer les coefficients de corrélation

        min_len = min(len(vallees_diff), len(pics_diff), len(transitions_diff))

        vallees_diff = vallees_diff[:min_len]
        pics_diff = pics_diff[:min_len]
        transitions_diff = transitions_diff[:min_len]

        corr_vallees_trans = np.corrcoef(vallees_diff, transitions_diff)[0, 1]
        corr_pics_trans = np.corrcoef(pics_diff, transitions_diff)[0, 1]


        # Effectuer une régression linéaire
        slope_vallees, intercept_vallees, r_vallees, p_vallees, std_err_vallees = linregress(vallees_diff, transitions_diff)
        slope_pics, intercept_pics, r_pics, p_pics, std_err_pics = linregress(pics_diff, transitions_diff)

        return {
            "Corrélation Vallées-Transitions": corr_vallees_trans,
            "Corrélation Pics-Transitions": corr_pics_trans,
            "Régression Vallées-Transitions": {
                "Pente": slope_vallees,
                "Intercept": intercept_vallees,
                "R-value": r_vallees,
                "P-value": p_vallees,
                "Erreur std": std_err_vallees
            },
            "Régression Pics-Transitions": {
                "Pente": slope_pics,
                "Intercept": intercept_pics,
                "R-value": r_pics,
                "P-value": p_pics,
                "Erreur std": std_err_pics
            }
        }
    else:
        return None

# Utilisation de la fonction
file_path = r'D:\\Documents\\Mémoire\\Data John Doe\\John Doe gaitway 3D locomotion_R16.txt'
output_file = r'D:\\Documents\\Mémoire\\Data John Doe\\ResultPython_JD_R16.txt'

print(f"Vérification de l'existence du fichier : {file_path}")
vallees, pics, transitions = read_and_process_file(file_path)

if vallees is not None and pics is not None and transitions is not None:
    write_results_to_file(vallees, pics, transitions, output_file)
    print(f"Les résultats ont été écrits dans le fichier : {output_file}")

    print(json.dumps(compare_differences(vallees, pics, transitions), indent=4))
