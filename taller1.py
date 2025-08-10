from pathlib import Path
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

df_Mo = importar_datos("Mo_unfiltered_10kV-50kV")
df_Rh = importar_datos("Rh_unfiltered_10kV-50kV")
df_W  = importar_datos("W_unfiltered_10kV-50kV")
