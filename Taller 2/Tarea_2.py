#Tarea 2 Métodos Computacionales II

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

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
#plt.plot(freqs, fourier_intensities)
#plt.title("Fourier transform of the signal")
#plt.ylabel("Intensity")
#plt.xlabel("Frequencies (Hz)")
#plt.savefig("1.a.pdf", bbox_inches="tight", pad_inches=0.1)

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

#plt.tight_layout()
#plt.show()
