import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Générer des données de test pour `speed` comme une combinaison de sinusoïdes pour plus de clarté
np.random.seed(0)
t = np.arange(1000)
speed = pd.Series(np.sin(0.02 * np.pi * t) + 0.5 * np.sin(0.05 * np.pi * t) + np.random.randn(1000) * 0.1)

# Calculer la moyenne mobile avec une fenêtre de 21
moving_average = speed.rolling(window=21, center=True).mean()

# Appliquer la transformation de Fourier
fft_result = np.fft.fft(moving_average.dropna())

# Fréquences associées
frequencies = np.fft.fftfreq(len(fft_result), d=(t[1] - t[0]))

# Prendre uniquement les fréquences positives
positive_freqs = frequencies[:len(frequencies)//2]
positive_amplitudes = np.abs(fft_result)[:len(frequencies)//2]

# Afficher les résultats
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Affichage des données originales et de la moyenne mobile
ax1.plot(t, speed, label='Données originales', alpha=0.5)
ax1.plot(t, moving_average, label='Moyenne mobile', linewidth=2)
ax1.set_title('Données originales et moyenne mobile')
ax1.legend()

# Affichage de la transformation de Fourier
ax2.plot(positive_freqs, positive_amplitudes)
ax2.set_title('Transformation de Fourier de la moyenne mobile')
ax2.set_xlabel('Fréquence')
ax2.set_ylabel('Amplitude')
ax2.set_xlim(0, 0.1)  # Limiter l'axe des X pour mieux voir les fréquences basses

plt.tight_layout()
plt.show()
