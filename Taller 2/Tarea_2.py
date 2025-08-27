#Tarea 2 Métodos Computacionales II

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy as sp
from matplotlib.colors import LogNorm
from scipy import ndimage as ndi
from PIL import Image
import pandas as pd
from astropy.timeseries import LombScargle

script_dir = os.path.dirname(os.path.abspath(__file__))
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

#3. Filtrando imágenes (FFT 2D)

#3.a. Desenfoque

ruta_imagen = os.path.join(script_dir, "gato Miette.jpg")
img = np.array(Image.open(ruta_imagen))

h, w, c = img.shape
X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
sigma = 0.1  # controla qué tanto desenfoque hay
gauss = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

img_fin = np.zeros_like(img)

for ch in range(3):
    F = np.fft.fft2(img[:, :, ch])
    F = np.fft.fftshift(F)
    
    F_fil = F * gauss
    
    F_ishift = np.fft.ifftshift(F_fil)
    desenfo = np.fft.ifft2(F_ishift).real
    
    desenfo = np.clip(desenfo, 0, 255)
    img_fin[:, :, ch] = desenfo.astype(np.uint8)

Image.fromarray(img_fin).save("3.a.jpg")
#3.b. Ruido periódico
#3.b.a. P_a_t_o
ruta_img2="p_a_t_o.jpg"
img2=np.array(Image.open(ruta_img2))
F = np.fft.fft2(img2)
F = np.fft.fftshift(F)

h, w = F.shape
X2, Y2 = np.meshgrid(np.arange(-w//2, w//2), np.arange(-h//2, h//2))
Z = np.hypot(X2, Y2) <= 20

#plt.imshow(abs(F),norm="log")
Z = (np.hypot(X2, Y2) > 3) & (np.hypot(X2, Y2) < 20)
F_filtered = F * (1 - Z)
#plt.figure(figsize=(6,7))
#plt.imshow(abs(F_filtered),norm="log")
img_filtrada = np.fft.ifft2(np.fft.fftshift(F_filtered))
Image.fromarray(img_filtrada.real).save("3.b.a.jpg")
#plt.imshow(img_filtrada.real)

#3.b.b. g_a_t_o
ruta_img3="g_a_t_o.png"
img3=np.array(Image.open(ruta_img3))
F2 = np.fft.fft2(img3)
F2 = np.fft.fftshift(F2)

h2, w2 = F2.shape
print(F2.shape)
X3, Y3 = np.meshgrid(np.arange(-w2//2, w2//2), np.arange(-h2//2, h2//2))
Z2 = (np.hypot(X3, Y3) > 15) & (np.hypot(X3, Y3) < 25)

F2_filtered = F2.copy()
F2_filtered[0:350, 370:380] = 0.
F2_filtered[400:759, 370:380] = 0.

F2_filtered[370:390,0:300] = 0.
F2_filtered[370:390,260:280] = 0.
img2_filtrada = np.fft.ifft2(np.fft.fftshift(F2_filtered))
Image.fromarray(img2_filtrada.real).save("3.b.a.jpg")

#5. Aplicación real: Reconstrucción tomográfica filtrada
file_path = os.path.join(script_dir, "4.npy")
data = np.load(file_path)

def reconstruccion_tomografica(image):
    
    shape = np.shape(image) #tamaño de la imagen 
    angles = np.arange(0,180,0.5) #Rotacion de los angulos por array 1D
    reconstruccion = np.zeros(shape) #array vacio para reconstruccion
    
    for (signal, angulo) in zip(image, angles): #iterar por 1D array y angulo
        imagen_rotada = ndi.rotate( # Rota la proyección para alinearla con el ángulo correspondiente
            np.tile(signal[:, None], shape[0]).T, # Replica la proyección 1D en 2D (replicando columnas)
            angulo,
            reshape=False, # Mantiene el mismo tamaño sin expandir la imagen al rotar
            mode="reflect") # Usa reflexión en los bordes para rellenar al rotar
        
        reconstruccion += imagen_rotada # Suma la proyección rotada a la imagen en reconstrucción
    return np.flipud(reconstruccion) # Devuelve la imagen reconstruida volteada verticalmente

rec1 = reconstruccion_tomografica(data)

def filtro(image):
    
    col = np.shape(image)[1]
    
    freq = np.fft.fftfreq(col) # Calcula las frecuencias correspondientes a la transformada de Fourier
    filter_freq = np.abs(freq) # Define el filtro como el valor absoluto de la frecuencia (filtro rampa)
    
    array_filtered = [] # Aplicar filtro a cada array
    for signal in image:
        signal_fft = np.fft.fft(signal) #FFT a cada array
        signal_filtrada = np.fft.ifft(signal_fft * filter_freq).real # Aplicar filtro a las frecuencias
        array_filtered.append(signal_filtrada)
    
    return array_filtered

data_filtered = filtro(data)
rec_data = reconstruccion_tomografica(filtro(data))

plt.imsave("5.png", rec_data, cmap="gray")

#NO NECESARIO

def espectro_image(image):

    resultados = []
    for signal in image:
        fft_signal = np.fft.fft(signal) #FFT del array 
        frecuencias = np.fft.fftfreq(len(signal)) # Calcula las frecuencias asociadas a la FFT
        intensidad = np.abs(fft_signal) # Obtiene la magnitud (intensidad) de las frecuencias
        
        espectro = frecuencias[frecuencias >= 0], intensidad[frecuencias >= 0]
        resultados.append(espectro)
    return resultados

FFT_data = espectro_image(data)
FFT_filtered = espectro_image(data_filtered)

fig = plt.figure(figsize=(8, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2])

ax1 = fig.add_subplot(gs[0, 0])  # Superior izquierdo
ax2 = fig.add_subplot(gs[0, 1])  # Superior derecho
ax3 = fig.add_subplot(gs[1, :])  # Inferior

ax1.imshow(rec1, cmap='gray', interpolation="nearest")
ax1.set_title('Tomografia reconstruida')
ax1.axis('off')

ax2.imshow(rec_data, cmap='gray', interpolation="nearest")
ax2.set_title('Tomografia datos filtrados')
ax2.axis('off')

for frecuencias, intensidades in FFT_data:
    ax3.plot(frecuencias, intensidades, alpha=0.3, c='blue', linewidth=0.8)
for frecuencias, intensidades in FFT_filtered:
    ax3.plot(frecuencias, intensidades, alpha=0.6, c='red', linewidth=1.2)

original_line = mlines.Line2D([], [], color='blue', alpha=0.7, linewidth=2, label='Datos originales')
filtered_line = mlines.Line2D([], [], color='red', alpha=0.7, linewidth=2, label='Datos filtrados')

ax3.set_yscale("log")
ax3.set_xscale("log")
ax3.set_xlabel('Frecuencia', fontsize=11, fontweight='bold')
ax3.set_ylabel('Intensidad (Amplitud)', fontsize=11, fontweight='bold')
ax3.set_title('Espectro de frecuencias', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(bottom=1e-2)
ax3.legend(handles=[original_line, filtered_line], loc='upper right', framealpha=0.9, fontsize=10)

plt.tight_layout(pad=1.5)
#plt.savefig("grafica_tomografia.png", dpi=300, bbox_inches="tight")
plt.close()

#4 Punto
df = pd.read_csv("OGLE-LMC-CEP-0001.dat", 
                 sep=" ", header=None, names=["tiempo", "brillo", "error"])

time = df["tiempo"].to_numpy()
brightness = df["brillo"].to_numpy()

frequency, power = LombScargle(time, brightness).autopower(
    minimum_frequency=0.01,maximum_frequency=10,samples_per_peak=len(time))

best_freq = frequency[np.argmax(power)] #esto es por el pico en el espacio de frecuencias, queremos la frecuencia que tiene ese pico
best_period = 1 / best_freq
ϕ = np.mod(best_freq * time, 1)

plt.figure(figsize=(6,4))
plt.plot(frequency, power, color="black")
plt.xlabel("Frecuencia [ciclos/día]")
plt.ylabel("Potencia")
plt.title("Espacio de frecuencias")
plt.grid(True)
plt.show()
#print(best_freq)
#print(best_period)

plt.figure(figsize=(6,4))
plt.scatter(ϕ , brightness, s=10,marker = "D", color="black")
plt.xlabel("Fase")
plt.ylabel("Brillo")
plt.title("Brillo vs fase")
plt.grid(True)
plt.savefig("4.pdf")
#plt.show()

# 4 Punto usando transformada rápida con linspace para obtener datos equiespaciados. 

t_uniforme = np.linspace(min(time), max(time), len(time))  
b_uniforme = np.interp(t_uniforme, time, brightness)

dt = np.mean(np.diff(time))
fft_vals = np.fft.fft(b_uniforme - np.mean(b_uniforme)) #En realidad el valor maximo es f=0, pero si no restamos todo se daña
fft_freqs = np.fft.fftfreq(len(t_uniforme), d=dt)  


pos = fft_freqs > 0
fft_freqs = fft_freqs[pos]
fft_power = (fft_vals[pos])


fmax = fft_freqs[np.argmax(fft_vals)]
#print( fmax)
ϕ = np.mod(fmax * time, 1)

plt.scatter(ϕ, brightness, s=10, alpha=0.7)
plt.grid(True)
plt.xlabel("Fase")
plt.ylabel("Brillo")
plt.title("Brillo de la estrella en función de la fase")
plt.savefig("4.pdf")
#plt.show()
