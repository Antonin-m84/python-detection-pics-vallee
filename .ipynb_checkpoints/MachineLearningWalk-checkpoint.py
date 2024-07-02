import numpy as np

# Charger les données du fichier texte
data = np.loadtxt('D:/Documents/Mémoire/Data John Doe/John Doe gaitway 3D locomotion_W7.txt')

# Extraire les colonnes temps et signal
time = data[:, 0]
signal = data[:, 19]

# Définir les états du modèle de Markov caché
states = ['normal', 'peak', 'valley']

# Définir les probabilités initiales des états
initial_prob = {'normal': 1.0, 'peak': 0.0, 'valley': 0.0}

# Définir les probabilités de transition entre les états
transition_prob = {
    'normal': {'normal': 0.8, 'peak': 0.2, 'valley': 0.0},
    'peak': {'normal': 0.0, 'peak': 0.0, 'valley': 1.0},
    'valley': {'normal': 1.0, 'peak': 0.0, 'valley': 0.0}
}

# Définir les probabilités d'émission basées sur le signal observé
# Pour simplifier, on utilisera des probabilités fixes pour chaque état
emission_prob = {
    'normal': lambda x: 1.0 if abs(x) < 0.5 else 0.0,
    'peak': lambda x: 1.0 if x > 1.0 else 0.0,
    'valley': lambda x: 1.0 if x < -1.0 else 0.0
}

# Fonction pour calculer la probabilité d'émission
def get_emission_prob(state, observation):
    return emission_prob[state](observation)

# Implémentation de l'algorithme de Viterbi
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialisation de la première observation
    for state in states:
        V[0][state] = start_p[state] * emit_p[state](obs[0])
        path[state] = [state]

    # Passer par chaque observation suivant la première
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for current_state in states:
            (prob, state) = max(
                (V[t - 1][prev_state] * trans_p[prev_state][current_state] * emit_p[current_state](obs[t]), prev_state)
                for prev_state in states
            )
            V[t][current_state] = prob
            new_path[current_state] = path[state] + [current_state]

        path = new_path

    # Trouver la probabilité maximale finale
    n = len(obs) - 1
    (prob, state) = max((V[n][final_state], final_state) for final_state in states)
    return (prob, path[state])

# Appliquer l'algorithme de Viterbi aux données du signal
observations = signal
prob, most_likely_states = viterbi(observations, states, initial_prob, transition_prob, emission_prob)

# Détecter les motifs de pics et de vallées
patterns = []
for i in range(len(most_likely_states) - 100):
    if most_likely_states[i] == 'peak' and 'valley' in most_likely_states[i:i+100]:
        patterns.append((time[i], signal[i]))

# Afficher les motifs détectés
print("Motifs détectés (temps, signal):")
for pattern in patterns:
    print(pattern)
