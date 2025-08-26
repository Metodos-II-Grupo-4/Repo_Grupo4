#Tarea 2 Métodos Computacionales II

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import matplotlib.lines as mlines
import os

#Aplicación real: Reconstrucción tomográfica filtrada
script_dir = os.path.dirname(os.path.abspath(__file__))
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