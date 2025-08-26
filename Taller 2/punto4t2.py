import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

df = pd.read_csv("OGLE-LMC-CEP-0001.dat", 
                 sep=" ", header=None, names=["tiempo", "brillo", "error"])

time = df["tiempo"].to_numpy()
brightness = df["brillo"].to_numpy()

frequency, power = LombScargle(time, brightness).autopower(
    minimum_frequency=0.01,maximum_frequency=0.5,samples_per_peak=500)

best_freq = frequency[np.argmax(power)] #esto es por el pico en el espacio de frecuencias
best_period = 1 / best_freq
ϕ = np.mod(best_freq * time, 1)

plt.figure(figsize=(6,4))
plt.plot(frequency, power, color="black")
plt.xlabel("Frecuencia [ciclos/día]")
plt.ylabel("Potencia")
plt.title("Espectro Lomb-Scargle")
plt.grid(True)
plt.show()

# Graficar curva de fase
plt.figure(figsize=(6,4))
plt.scatter(ϕ , brightness, s=10,marker = "D", color="black")
plt.xlabel("Fase")
plt.ylabel("Brillo")
plt.title("Brillo vs fase con Lomb–Scargle")
plt.grid(True)
plt.savefig("4.pdf")
plt.show()