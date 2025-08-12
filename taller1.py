from pathlib import Path
from scipy.signal import find_peaks, peak_widths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

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
print(df_Mo)
#1. Reconocimiento
def graficar_con_colorbar(df, ax, titulo, cmap):
    voltajes = df["voltaje_kV"].unique()
    norm = mpl.colors.Normalize(vmin=min(voltajes), vmax=max(voltajes))
    
    for v in voltajes:
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
graficar_con_colorbar(df_Rh, axes[0], "Rh", plt.cm.Reds)
graficar_con_colorbar(df_Mo, axes[1], "Mo", plt.cm.Blues)
graficar_con_colorbar(df_W,  axes[2], "W", plt.cm.Greens)

plt.tight_layout()
plt.savefig("1.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()

#VOLTAJE
V = 30

#2a. Remover los picos
def remover_picos(df, ax, titulo, Elim=15, V=31, thrd = 0.25, N=2):
    subset = df[df["voltaje_kV"] == V]
    subset_E = subset[subset["energy_keV"] >= Elim].reset_index()
    
    peaks, _ = find_peaks(subset_E["fluence"], height=0)
    peaks_filtered = [p for p in peaks if subset_E.loc[p, "fluence"] > subset_E["fluence"].max() * thrd]
    
    indices = set()
    for p in peaks_filtered:
        indices.update(range(max(0, p-N), min(len(subset), p+N+1)))
    
    peak = subset_E.loc[sorted(indices), ["energy_keV", "fluence"]]
    no_peaks = subset[(~subset['energy_keV'].isin(peak['energy_keV'])) & (~subset['fluence'].isin(peak['fluence']))]
    
    ax.set_title(f'{titulo} (V = {V}kV)')
    ax.set_xlabel("Energía (keV)")
    ax.set_ylabel(r"Fluencia  keV$^{-1}$ cm$^{-2}$")
    ax.plot(subset["energy_keV"], subset["fluence"], 'k-', marker='o', ms=3, label="Original")
    ax.plot(no_peaks["energy_keV"], no_peaks["fluence"], 'b--', label="Sin picos")
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
def interpolacion(df, ax, title):
    spl = UnivariateSpline(df["energy_keV"], df["fluence"], s=1.5)
    f_interp = interp1d(df["energy_keV"], df["fluence"], kind='cubic')
    
    X = np.linspace(df["energy_keV"].min(), df["energy_keV"].max(), 200)
    Y = f_interp(X)
    Y_spl = spl(X)
    
    ax.set_title(f'Interpolación {title}')
    ax.set_xlabel("Energía (keV)")
    ax.set_ylabel(r"Fluencia  keV$^{-1}$ cm$^{-2}$")
    ax.scatter(df["energy_keV"], df["fluence"], color="red", s=6, label="Datos originales")
    ax.plot(X, Y, label="Interpolación cúbica", color="green")
    ax.plot(X, Y_spl, 'b--', label="Ajuste suavizado")
    ax.legend()

    return(X, Y, Y_spl)

fig, axes = plt.subplots(3, 1, figsize=(8, 8))

X_Rh,Y_Rh,Y_spl_Rh=interpolacion(no_peak_Rh, axes[0], "Rh")
X_Mo,Y_Mo,Y_spl_Mo=interpolacion(no_peak_Mo, axes[1], "Mo")
X_W,Y_W,Y_spl_W=interpolacion(no_peak_W, axes[2], "W")

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
    
    return (x[left_idx], x[right_idx]),(half_max,half_max), fwhm_value
#DESDE AQUÍ LA CORRECCIÓN
maximos_Rh = []
maximos_Mo = []
maximos_W = []

maximosE_Rh = []
maximosE_Mo = []
maximosE_W = []

fwhm_Rh = []
fwhm_Mo = []
fwhm_W = []

Voltaje = []

for V in range(10, 51):
    Voltaje.append(V)
    
    subset_Mo = df_Mo[df_Mo["voltaje_kV"] == V]
    spl = UnivariateSpline(subset_Mo["energy_keV"], subset_Mo["fluence"], s=1.5)
    X = np.linspace(subset_Mo["energy_keV"].min(), subset_Mo["energy_keV"].max(), 200)
    Y_spl = spl(X)
    max_y, max_E = maximos(X, Y_spl)
    maximos_Mo.append(max_y)
    maximosE_Mo.append(max_E)
    fwhm_Mo.append(fwhm(X, Y_spl)[2])
    
    subset_Rh = df_Rh[df_Rh["voltaje_kV"] == V]
    spl = UnivariateSpline(subset_Rh["energy_keV"], subset_Rh["fluence"], s=1.5)
    X = np.linspace(subset_Rh["energy_keV"].min(), subset_Rh["energy_keV"].max(), 200)
    Y_spl = spl(X)
    max_y, max_E = maximos(X, Y_spl)
    maximos_Rh.append(max_y)
    maximosE_Rh.append(max_E)
    fwhm_Rh.append(fwhm(X, Y_spl)[2])
    
    subset_W = df_W[df_W["voltaje_kV"] == V]
    spl = UnivariateSpline(subset_W["energy_keV"], subset_W["fluence"], s=1.5)
    X = np.linspace(subset_W["energy_keV"].min(), subset_W["energy_keV"].max(), 200)
    Y_spl = spl(X)
    max_y, max_E = maximos(X, Y_spl)
    maximos_W.append(max_y)
    maximosE_W.append(max_E)
    fwhm_W.append(fwhm(X, Y_spl)[2]) 

fig, axes = plt.subplots(4, 1, figsize=(8, 8))
#GRAFICAMOS MAXIMO CONTINUO EN FUNCION DE V
axes[0].plot(Voltaje, maximos_Rh, label="Rh", color="blue")
axes[0].plot(Voltaje, maximos_Mo, label="Mo", color="red")
axes[0].plot(Voltaje, maximos_W, label="W", color="yellow")
axes[0].set_title(f"Maximo continuo (Fluencia) en función de voltaje (V)")
axes[0].set_xlabel("V (V)")
axes[0].set_ylabel(r"Fluencia  keV$^{-1}$ cm$^{-2}$")
axes[0].legend()

#GRAFICAMOS ENERGÍA DEL MAXIMO EN FUNCION DE V
axes[1].plot(Voltaje, maximosE_Rh, label="Rh", color="blue")
axes[1].plot(Voltaje, maximosE_Mo, label="Mo", color="red")
axes[1].plot(Voltaje, maximosE_W, label="W", color="yellow")
axes[1].set_title(f"Energía del maximo continuo (E) en función de voltaje (V)")
axes[1].set_xlabel("V (V)")
axes[1].set_ylabel("E (keV)")
axes[1].legend()

#GRAFICAMOS FWHM EN FUNCION DE V
axes[2].plot(Voltaje, fwhm_Rh, label="Rh", color="blue")
axes[2].plot(Voltaje, fwhm_Mo, label="Mo", color="red")
axes[2].plot(Voltaje, fwhm_W, label="W", color="yellow")
axes[2].set_title(f"FWHM en función de voltaje (V)")
axes[2].set_xlabel("V (V)")
axes[2].set_ylabel("FWHM (keV)")
axes[2].legend()

#GRAFICAMOS MAXIMO CON RESPECTO A ENERGIA DEL MAXIMO
axes[3].plot(maximosE_Rh, maximos_Rh, label="Rh", color="blue")
axes[3].plot(maximosE_Mo, maximos_Mo, label="Mo", color="red")
axes[3].plot(maximosE_W, maximos_W, label="W", color="yellow")
axes[3].set_title("Máximo (Fluencia) en función de energía del máximo (E)")
axes[3].set_xlabel("E (keV)")
axes[3].set_ylabel(r"Fluencia  keV$^{-1}$ cm$^{-2}$")
axes[3].legend()

plt.tight_layout()
plt.savefig("2.c.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()