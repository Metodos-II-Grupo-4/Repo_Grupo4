from pathlib import Path
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
#plt.show()

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
no_peak_Rh = remover_picos(df_Rh, axes[0], "Rh", V=30)
no_peak_Mo = remover_picos(df_Mo, axes[1], "Mo", V=30, thrd=0.2)
no_peak_W = remover_picos(df_W,  axes[2], "W", V=30, Elim=5)

plt.tight_layout()
plt.savefig("2.a.pdf", bbox_inches="tight", pad_inches=0.1)
#plt.show()

