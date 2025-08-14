from pathlib import Path
from scipy.signal import find_peaks, peak_widths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RBFInterpolator

#CARGAR DATOS

def importar_datos(carpeta):
    carpeta_path = Path(carpeta)
    datos = []
    for archivo in carpeta_path.iterdir():
        if archivo.suffix.lower() == ".dat":
            
            voltaje = int(''.join(filter(str.isdigit, archivo.stem)))
            df = pd.read_csv(
                archivo,
                sep = r"\s+",
                comment = "#",
                header = None,
                names = ["energy_keV", "fluence"],
                encoding = "latin1"
            )
            df["voltaje_kV"] = voltaje
            datos.append(df)
    return pd.concat(datos, ignore_index=True)

#DATAFRAMES
df_Mo = importar_datos("Mo_unfiltered_10kV-50kV")
df_Rh = importar_datos("Rh_unfiltered_10kV-50kV")
df_W  = importar_datos("W_unfiltered_10kV-50kV")
#1. Reconocimiento
def graficar_con_colorbar(df, ax, titulo, cmap):
    voltajes = df["voltaje_kV"].unique()
    norm = mpl.colors.Normalize(vmin=min(voltajes), vmax=max(voltajes))
    
    for v in voltajes[::5]:
        subset = df[df["voltaje_kV"] == v]
        ax.plot(subset["energy_keV"], subset["fluence"], color=cmap(norm(v)))
    
    ax.set_title(titulo)
    ax.set_xlabel("Energía (keV)")
    ax.set_ylabel(r"Fluencia  keV$^{-1}$ cm$^{-2}$")
    
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Voltaje (kV)")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
graficar_con_colorbar(df_Rh, axes[0], "Rh", plt.cm.viridis)
graficar_con_colorbar(df_Mo, axes[1], "Mo", plt.cm.viridis)
graficar_con_colorbar(df_W,  axes[2], "W", plt.cm.viridis)

plt.tight_layout()
plt.savefig("1.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()

#VOLTAJE
V = 30

#2a. Remover los picos
def remover_picos(df, ax, titulo, Elim=15, V=31, thrd = 0.25, N=2, plot=True):
    subset = df[df["voltaje_kV"] == V]
    subset_E = subset[subset["energy_keV"] >= Elim].reset_index()
    
    peaks, _ = find_peaks(subset_E["fluence"], height=0)
    peaks_filtered = [p for p in peaks if subset_E.loc[p, "fluence"] > subset_E["fluence"].max() * thrd]
    
    indices = set()
    for p in peaks_filtered:
        indices.update(range(max(0, p-N), min(len(subset), p+N+1)))
    
    peak = subset_E.loc[sorted(indices), ["energy_keV", "fluence"]]
    no_peaks = subset[(~subset['energy_keV'].isin(peak['energy_keV'])) & (~subset['fluence'].isin(peak['fluence']))]
    if plot:
        ax.set_title(f'{titulo} (V = {V}kV)')
        ax.set_xlabel("Energía (keV)")
        ax.set_ylabel(r"Fluencia  keV$^{-1}$ cm$^{-2}$")
        ax.plot(subset["energy_keV"], subset["fluence"], 'k-', marker='o', ms=3, label="Original")
        ax.scatter(peak["energy_keV"], peak["fluence"], color="red", label="Eliminados", zorder=5)
        ax.legend()
    
    return no_peaks
    
fig, axes = plt.subplots(3, 1, figsize=(8, 8))
no_peak_Rh = remover_picos(df_Rh, axes[0], "Rh", V=V)
no_peak_Mo = remover_picos(df_Mo, axes[1], "Mo", V=V, thrd=0.2)
no_peak_W = remover_picos(df_W,  axes[2], "W", V=V, Elim=5)

plt.tight_layout()
plt.savefig("2.a.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()

#2b
def interpolacion(df, ax, title, plot=True):
    X = df["energy_keV"].to_numpy().reshape(-1, 1)
    Y = df["fluence"]
    rbf = RBFInterpolator(X, Y, smoothing=20)
    X_correcto = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    Y_correcto = rbf(X_correcto)
    if plot:
        ax.set_title(f'Interpolación {title}')
        ax.set_xlabel("Energía (keV)")
        ax.set_ylabel(r"Fluencia  keV$^{-1}$ cm$^{-2}$")
        ax.scatter(df["energy_keV"], df["fluence"], color="red", s=6, label="Datos originales")
        ax.plot(X_correcto, Y_correcto, 'b--', label="Ajuste suavizado")
        ax.legend()

    return X_correcto, Y_correcto

fig, axes = plt.subplots(3, 1, figsize=(8, 8))

X_Rh, Y_Rh = interpolacion(no_peak_Rh, axes[0], "Rh")
X_Mo, Y_Mo = interpolacion(no_peak_Mo, axes[1], "Mo")
X_W, Y_W = interpolacion(no_peak_W, axes[2], "W")

plt.tight_layout()
plt.savefig("2.b.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()

#2c
def maximos(x, y):    
    dx = (x[-1] - x[0]) / len(x)
    yp = np.zeros_like(x)

    for i in range(len(x)):
        if i == 0:
            yp[i] = (y[i+1] - y[i]) / dx
        elif i == len(x) - 1:
            yp[i] = (y[i] - y[i-1]) / dx
        else:
            yp[i] = (y[i+1] - y[i-1]) / (2*dx)

    maximos_condicion = (yp[:-1] > 0) & (yp[1:] < 0)

    maximos_indices = np.where(maximos_condicion)[0] + 1
    maximos_x = x[maximos_indices]
    maximos_y = y[maximos_indices]

    idx_max_global = np.argmax(maximos_y)
    return (maximos_x[idx_max_global], maximos_y[idx_max_global])

def fwhm(x, y):
    max_x, max_y = maximos(x, y)
    half_max = max_y / 2

    left_idx = np.where(y[:np.argmax(y)] < half_max)[0][-1]
    right_idx = np.where(y[np.argmax(y):] < half_max)[0][0] + np.argmax(y)

    fwhm_value = x[right_idx] - x[left_idx]
    
    return fwhm_value

#2c

#Se crean listas vacías para cada elemento
voltajes_Rh, max_val_Rh, max_ener_Rh, fwhm_Rh = [], [], [], []
voltajes_Mo, max_val_Mo, max_ener_Mo, fwhm_Mo = [], [], [], []
voltajes_W,  max_val_W,  max_ener_W,  fwhm_W  = [], [], [], []

#Función para procesar un elemento
def analizar_elemento(df, thrd, Elim, voltajes_lista, max_val_lista, max_ener_lista, fwhm_lista):
    
    voltajes_unicos = sorted(df["voltaje_kV"].unique())
    
    for V in voltajes_unicos:
        
        # Remover picos
        no_peaks = remover_picos(df, plt.gca(), "", V=V, thrd=thrd, Elim=Elim, plot=False)  # sin título
        # Interpolación del continuo
        X_corr, Y_corr = interpolacion(no_peaks, plt.gca(), "", plot=False)  # sin título
        X_corr = X_corr.flatten()

        # Calcular máximo y energía del máximo
        ener_max, val_max = maximos(X_corr, Y_corr)
        # Calcular FWHM
        ancho_fwhm = fwhm(X_corr, Y_corr)

        # Guardar resultados
        voltajes_lista.append(V)
        max_val_lista.append(val_max)
        max_ener_lista.append(ener_max)
        fwhm_lista.append(ancho_fwhm)

#Usar la función en los dataframes ORIGINALES
analizar_elemento(df_Rh, 0.25, 15, voltajes_Rh, max_val_Rh, max_ener_Rh, fwhm_Rh)
analizar_elemento(df_Mo, 0.20, 15, voltajes_Mo, max_val_Mo, max_ener_Mo, fwhm_Mo)
analizar_elemento(df_W,  0.25, 5,  voltajes_W,  max_val_W, max_ener_W,  fwhm_W)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

#Máximo vs voltaje
axs[0,0].plot(voltajes_Rh, max_val_Rh, label="Rh")
axs[0,0].plot(voltajes_Mo, max_val_Mo, label="Mo")
axs[0,0].plot(voltajes_W,  max_val_W, label="W")
axs[0,0].set_xlabel("Voltaje (kV)")
axs[0,0].set_ylabel("Máximo continuo")
axs[0,0].legend()

#Energía del máximo vs voltaje
axs[0,1].plot(voltajes_Rh, max_ener_Rh, label="Rh")
axs[0,1].plot(voltajes_Mo, max_ener_Mo, label="Mo")
axs[0,1].plot(voltajes_W,  max_ener_W, label="W")
axs[0,1].set_xlabel("Voltaje (kV)")
axs[0,1].set_ylabel("Energía del máximo (keV)")
axs[0,1].legend()

#FWHM vs voltaje
axs[1,0].plot(voltajes_Rh, fwhm_Rh, label="Rh")
axs[1,0].plot(voltajes_Mo, fwhm_Mo, label="Mo")
axs[1,0].plot(voltajes_W,  fwhm_W, label="W")
axs[1,0].set_xlabel("Voltaje (kV)")
axs[1,0].set_ylabel("FWHM (keV)")
axs[1,0].legend()

#Máximo vs energía del máximo
axs[1,1].plot(max_ener_Rh, max_val_Rh, label="Rh")
axs[1,1].plot(max_ener_Mo, max_val_Mo, label="Mo")
axs[1,1].plot(max_ener_W,  max_val_W, label="W")
axs[1,1].set_xlabel("Energía del máximo (keV)")
axs[1,1].set_ylabel("Máximo continuo")
axs[1,1].legend()

plt.tight_layout()
plt.savefig("2.c.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()
