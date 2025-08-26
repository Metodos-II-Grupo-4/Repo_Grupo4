#Tarea 2 Métodos Computacionales II

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

#Punto 1

#Definimos la señal
def generate_data(tmax,dt,A,freq,noise):
    '''Generates a sin wave of the given amplitude (A) and frequency (freq),
    sampled at times going from t=0 to t=tmax, taking data each dt units of time.
    A random number with the given standard deviation (noise) is added to each data point.
    ----------------
    Returns an array with the times and the measurements of the signal. '''
    ts = np.arange(0,tmax+dt,dt)
    return ts, np.random.normal(loc=A*np.sin(2*np.pi*ts*freq),scale=noise)

#1.a.a.
#Definimos nuestra transformada de Fourier
def Fourier_transform(t, y, f):
    Fourier = []
    for freq in f:
        Fourier.append(np.sum(y*np.exp(-2*np.pi*1j*freq*t)))
    return np.array(Fourier)

#1.a.b.
#Generamos la señal
señal = generate_data(1, 0.001, 1, 50, 0.5)
t_data, y_data = señal
freqs = np.arange(0, 1350, 0.1)
fourier_intensities = np.abs(Fourier_transform(t_data, y_data, freqs))
plt.plot(freqs, fourier_intensities)
plt.title("Fourier transform of the signal")
plt.ylabel("Intensity")
plt.xlabel("Frequencies (Hz)")
plt.savefig("1.a.pdf", bbox_inches="tight", pad_inches=0.1)

#1.b.
#Usamos la función del taller 1 para remover picos
def remover_picos(tf, Elim=40, thrd=0.25, N=2):
    no_peaks = tf.copy() 
    # Subconjunto desde Elim en adelante
    subset_tf = tf[Elim:]
    # Encontrar picos
    peaks, _ = sp.signal.find_peaks(subset_tf, height=0)
    # Filtrar picos que superen el umbral relativo
    peaks_filtered = [p for p in peaks if subset_tf[p] > subset_tf.max() * thrd]
    # Eliminar pico y vecinos
    for p in peaks_filtered:
        start = max(0, p-N)
        end = min(len(subset_tf), p+N+1)
        no_peaks[Elim+start:Elim+end] = 0 
    return no_peaks

#Generamos muchos conjuntos con diferentes S-N
SN_t = np.logspace(-2, 0, 100)
frequency = 50
A = SN_t * frequency
SN_f = []
for amplitud in A:
    señal = generate_data(1, 0.001, amplitud, frequency, 0.5)
    t_data, y_data = señal
    freqs = np.arange(0, 1350, 0.1)
    fourier_intensities = np.abs(np.fft.fft(y_data)) #Usamos FFT
    max_intensity = np.max(fourier_intensities)
    noise_only = remover_picos(fourier_intensities)
    std = np.std(noise_only)
    SN_f.append(max_intensity/std)
#plt.figure()
#plt.loglog(SN_t, SN_f)
#plt.title("SN_freq vs. SN_time")
#plt.xlabel("SN_time")
#plt.ylabel("SN_freq")
#plt.show()

#1.c. 
#Reciclamos las funciones auxiliares desde la tarea 1
def maximos(x, y):    
    dx = x[1] - x[0] #Paso en x
    yp = np.gradient(y, dx) #Derivada numérica

    #El cambio de derivada es la condición del maximo
    maximos_condicion = (yp[:-1] > 0) & (yp[1:] < 0)
    maximos_indices = np.where(maximos_condicion)[0] + 1

    #Filtrar los que están fuera del rango válido
    maximos_indices = maximos_indices[maximos_indices < len(x)]

    if len(maximos_indices) == 0:
        #Si no hay máximos, devolver el máximo global
        idx_max = np.argmax(y)
        return (x[idx_max], y[idx_max])

    maximos_x = x[maximos_indices]
    maximos_y = y[maximos_indices]
    #Escoger el máximo global de entre los locales
    idx_max_global = np.argmax(maximos_y)
    return (maximos_x[idx_max_global], maximos_y[idx_max_global])




def fwhm(x, y):
    max_x, max_y = maximos(x, y)
    half_max = max_y / 2

    left_idx = np.where(y[:np.argmax(y)] < half_max)[0][-1]
    right_idx = np.where(y[np.argmax(y):] < half_max)[0][0] + np.argmax(y)

    fwhm_value = x[right_idx] - x[left_idx]
    
    return fwhm_value

#Generamos la señal
T_max = np.arange(1, 200, 1)
anchos = []
for t_max in T_max:
    señal = generate_data(t_max, 0.001, 1, 50, 0.5)
    t_data, y_data = señal
    fft_vals = np.fft.fft(y_data)
    freqs = np.fft.fftfreq(len(y_data), d=0.001) #d=dt 
    mask = freqs >= 0
    freqs = freqs[mask]
    fft_vals = np.abs(fft_vals)[mask]
    anchos.append(fwhm(freqs, fft_vals))
#plt.figure()
#plt.plot(T_max, anchos)
#plt.title("FWHM Dependency on observation time")
#plt.ylabel("FWHM")
#plt.xlabel("Observation time (s)")

#1. d. BONO
def generate_data(tmax,dt,A,freq,noise, sampling_noise=0):
    '''Generates a sin wave of the given amplitude (A) and frequency (freq),
    sampled at times going from t=0 to t=tmax, taking data each dt units of time.
    A random number with the given standard deviation (noise) is added to each data point.
    ----------------
    Returns an array with the times and the measurements of the signal. '''
    ts = np.arange(0,tmax+dt,dt)
    if sampling_noise > 0:
        ts = np.random.normal(loc=ts, scale=sampling_noise)
    return ts, np.random.normal(loc=A*np.sin(2*np.pi*ts*freq),scale=noise)

def Fourier_transform(t, y, f):
    dt = np.diff(t, prepend=t[0])
    Fourier = []
    for freq in f:
        Fourier.append(np.sum(y * np.exp(-2j*np.pi*freq*t) * dt))
    return np.array(Fourier)

sampling_noises = [0, 0.0001, 0.0002, 0.0005]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for i, sampling_noise in enumerate(sampling_noises):
    señal = generate_data(1, 0.001, 1, 50, 0.2, sampling_noise=sampling_noise)
    t_data, y_data = señal
    freqs = np.arange(0, 1350, 0.5)
    fourier_intensities = np.abs(Fourier_transform(t_data, y_data, freqs))
    
    axes[i].plot(freqs, fourier_intensities)
    axes[i].set_title(f"Sampling noise = {round(sampling_noise*100/0.001, 4)}% of dt")
    axes[i].set_xlabel("Frecuencia (Hz)")
    axes[i].set_ylabel("Intensidad Fourier")

plt.tight_layout()
plt.savefig("1.d (BONO).pdf", bbox_inches="tight", pad_inches=0.1)

#Punto 2

#2. a. Arreglo
datos_2 = pd.read_csv("Taller 2/SN_d_tot_V2.0.csv")
years = datos_2["year"]
spots = datos_2["spots"].copy().astype(float)
spots_without_error = spots[spots != -1]
indices = np.array(spots_without_error.index)
spots_without_error = np.array(spots_without_error)
spl = sp.interpolate.CubicSpline(indices, spots_without_error)

#Reemplazamos los spots == -1 por los de interpolación:
indices_to_replace = spots[spots == -1].index
spots[indices_to_replace] = spl(indices_to_replace)
datos_2["spots"] = spots

#2. b.
#Para hallar la frecuencia (y por tanto el periodo) con mayor precisión, aumentamos el tiempo de observación con ceros.

spots = spots - np.mean(spots)
ft = np.fft.fft(spots, n=len(spots)*2) #Zero padding al doble (BONO)
ft_freqs = np.fft.fftfreq(len(ft), d=1)
mask = (ft_freqs >= 0)
ft_abs = np.abs(ft[mask])
ft_freqs = ft_freqs[mask]
periodo = 1/ft_freqs[np.argmax(ft_abs[1:])+1]
np.savetxt("2.b.txt", np.array([periodo]))

#---------------------------------------------------------------------------------------------------------------------
ft_full = np.fft.fft(spots, n=len(spots)*2)
ft_freqs_full = np.fft.fftfreq(len(ft_full), d=1)

# Filtro gaussiano pasa bajas completo
lowpass_sigma_factor = 5.0
sigma_low = lowpass_sigma_factor * 1/7300
H_low_full = np.exp(-(ft_freqs_full**2) / (2 * sigma_low**2))

# Filtrar todo el espectro
ft_filtered_full = ft_full * H_low_full
y_filtered_full = np.fft.ifft(ft_filtered_full).real

plt.figure(figsize=(10,5))
plt.plot(years, spots, label="Original", alpha=0.5)
plt.plot(years, y_filtered_full[:75818], label="Filtrada (dominio freq)", color="red", linewidth=2)
plt.xlabel("Año")
plt.ylabel("Número de manchas")
plt.title("Original vs Filtrada (filtro gaussiano en frecuencia)")
plt.legend()
plt.tight_layout()
plt.savefig("2.b.data.pdf", bbox_inches="tight", pad_inches=0.1)

#---------------------------------------------------------------------------------------------------------------------

local_maxima_indices = sp.signal.argrelextrema(y_filtered_full[:75818], np.greater)
fechas = years[local_maxima_indices[0]]
maxima = y_filtered_full[local_maxima_indices[0]]
plt.figure()
plt.plot(fechas, maxima)
plt.xlabel("Año")
plt.ylabel("Valor máximo (spots)")
plt.title("Maximo vs. año")
plt.savefig("2.b.maxima.pdf", bbox_inches="tight", pad_inches=0.1)