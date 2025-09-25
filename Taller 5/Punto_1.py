import numpy as np
import matplotlib.pyplot as plt
from numba import njit

N = 150
J = 1
beta = 1/2
epocas = 2000000
seed = 42
spins = np.random.choice([-1, 1], size=(N, N))

rng = np.random.default_rng(seed) #Retorna un generador random con la semilla de arriba.

def Vecinos_cercanos_sum(spins, i, j):
    N = spins.shape[0]
    return (spins[(i+1) % N, j] + spins[(i-1) % N, j] +
            spins[i, (j+1) % N] + spins[i, (j-1) % N])

def Energia_total(spins):
  E = 0
  Nloc = spins.shape[0]
  for i in range(Nloc):
    for j in range(Nloc):
      E += -J * spins[i,j] * Vecinos_cercanos_sum(spins, i, j)
  return E/2

# Guardamos el estado inicial
snaps = {}
snaps[0] = spins.copy() 

E = Energia_total(spins)
M = np.sum(spins)
#Ahora si, Metropolis.
aceptados = 0
energias_normalizadas = np.zeros(epocas)
magnetizaciones_normalizadas = np.zeros(epocas)
for epoca in range(epocas):
#Paso 1. Perturbar el sitema, es decir, cambiar de sentido un solo espín en una posición i, j elegida aleatoriamente.
  i = rng.integers(N)
  j = rng.integers(N)
#Paso 2. Calcular la nueva energía con ese espín perturbado Enew, y calcular la diferencia de energía deltaE
  deltaE = 2 * J * spins[i,j] * Vecinos_cercanos_sum(spins, i, j)
  s = spins[i, j]
#Paso 3. Si deltaE <= 0, acepta la nueva configuración.
#Paso 4. De lo contrario, lanzar un número aleatorio u, uniforme de 0 a 1, y comparar Si u <= exp(−\beta deltaE), aceptar la nueva configuración. De lo contrario, dejar la configuración vieja.
  if deltaE <= 0 or rng.random() < np.exp(-beta * deltaE):
    spins[i, j] = -s
    E += deltaE
    M += -2*s
    aceptados += 1
  energias_normalizadas[epoca] = E/(4*N**2)
  magnetizaciones_normalizadas[epoca] = M/(N**2)

snaps[1] = spins.copy()  #Guardamos estado final

# Creamos la figura con 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18,5))

# 1) Evolución de energía y magnetización
axes[1].plot(np.arange(epocas), energias_normalizadas, label="Energía normalizada", color="k", lw=0.8)
axes[1].plot(np.arange(epocas), magnetizaciones_normalizadas, label="Magnetización por espín", color="red", lw=0.8)
axes[1].set_xlabel("Épocas")
axes[1].set_ylabel("Valor normalizado")
axes[1].set_title(f"Energía y Magnetización")
axes[1].legend()

# 2) Espines antes
axes[0].imshow(snaps[0], cmap="coolwarm", vmin=-1, vmax=1)
axes[0].set_title("Antes")
axes[0].axis("off")

# 3) Espines después
axes[2].imshow(snaps[1], cmap="coolwarm", vmin=-1, vmax=1)
axes[2].set_title("Después")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("1.a.pdf")
plt.close()

def tamano_promedio_dominios(spins):
    N = spins.shape[0]
    visited = np.zeros_like(spins, dtype=bool)
    sizes = []
    for i in range(N):
        for j in range(N):
            if visited[i,j]: 
                continue
            val = spins[i,j]
            q = [(i,j)]
            visited[i,j] = True
            sz = 0
            while q:
                x,y = q.pop()
                sz += 1
                for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = (x+dx) % N, (y+dy) % N
                    if not visited[nx,ny] and spins[nx,ny] == val:
                        visited[nx,ny] = True
                        q.append((nx,ny))
            sizes.append(sz)
    return np.mean(sizes)

promedio_spins = tamano_promedio_dominios(spins)
np.savetxt("BONO.1.a.txt", promedio_spins)

#1b
@njit
def sweep_metropolis(spins, beta, J):
    N = spins.shape[0]
    for _ in range(N*N):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s = spins[i, j]
        Snn = (spins[(i+1)%N, j] + spins[(i-1)%N, j] +
               spins[i, (j+1)%N] + spins[i, (j-1)%N])
        dH = 2.0 * J * s * Snn
        if dH <= 0.0 or np.random.random() < np.exp(-beta*dH):
            spins[i, j] = -s

N = 40
J = 1.0
betas = np.linspace(0.05, 0.95, 90)
Equilibrio = 500 # Sweeps para equilibrar
Medicion = 1500 # Sweeps para medir
int_muestreo = 20 #Intervalo de muestreo
rng = np.random.default_rng(1234)

# Simulación para varios β
Cv_list = []
spins = rng.choice([-1, 1], size=(N, N))

for beta in betas:
    # Equilibrar
    for _ in range(Equilibrio):
        sweep_metropolis(spins, beta, J)

    # Medición
    muestras_E = []
    for paso in range(1, Medicion+1):
        sweep_metropolis(spins, beta, J)
        if paso % int_muestreo == 0:
            muestras_E.append(Energia_total(spins, J))

    # Mismo algoritmo que antes, solo cambiamos el observable
    muestras_E = np.array(muestras_E)
    E_mean = muestras_E.mean()
    E2_mean = (muestras_E**2).mean()
    Cv = (beta**2/N**2)* (E2_mean - E_mean**2)
    Cv_list.append(Cv)

# Gráfico
beta_c = 0.5*np.log(1+np.sqrt(2))   
plt.figure(figsize=(6,4))
plt.plot(betas, Cv_list, lw=2, color='k')
plt.axvline(beta_c, color='r', linestyle='-', alpha=0.7, label='β crítico (teoría)')
plt.xlabel("Thermodynamic β ")
plt.ylabel("Specific Heat from simulation")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("1.b.pdf", bbox_inches="tight")