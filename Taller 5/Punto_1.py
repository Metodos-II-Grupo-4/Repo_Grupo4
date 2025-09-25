import numpy as np
import matplotlib.pyplot as plt
from numba import njit

N = 500
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

plt.figure(figsize=(8,5))
plt.plot(np.arange(epocas), energias_normalizadas, label="Energía normalizada", color="k")
plt.plot(np.arange(epocas), magnetizaciones_normalizadas, label="Magnetización por espín", color="red")
plt.xlabel("Epocas")
plt.ylabel("Valor normalizado")
plt.title(f"Ising Metropolis (N={N}, β={beta}, pasos={epocas:,})")
plt.legend()
plt.tight_layout()
plt.savefig("1.a.pdf")

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
N = 40
betas = np.concatenate([
    np.linspace(0.05, 0.30, 6, endpoint=False),
    np.linspace(0.30, 0.60, 11, endpoint=False),  # menos puntos
    np.linspace(0.60, 0.95, 4)
])

Equlibrio   = 100   
Medicion    = 300   
int_muestro = 30 

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

@njit
def energia_norm_hamilton(spins, J):
    N = spins.shape[0]
    E = 0.0
    for i in range(N):
        for j in range(N):
            E -= J * spins[i,j] * (spins[i,(j+1)%N] + spins[(i+1)%N,j])  # enlace una vez
    return E / (2.0 * N * N)  # equivale a ε/(4N^2)

def run_beta(spins, beta, J, aleatorio, Equlibrio_local=1000, Medicion_local=2000, int_muestro_local=10):
    for _ in range(Equlibrio_local):
        sweep_metropolis(spins, beta, J)

    muestras_E = []
    for e in range(1, Medicion_local+1):
        sweep_metropolis(spins, beta, J)
        if e % int_muestro_local == 0:
            muestras_E.append(energia_norm_hamilton(spins, J))

    muestras_E = np.asarray(muestras_E, dtype=float)
    E_mean = muestras_E.mean()
    E2_mean = (muestras_E**2).mean()
    Cv = (beta**2) * (N**2) * (E2_mean - E_mean**2)
    return Cv, E_mean, spins

Cv_list = []
Emean_list = []
for i, beta in enumerate(betas):
    equil = Equlibrio if i > 0 else 2 * Equlibrio  
    Cv, Emean, spins = run_beta(
        spins, beta, J, aleatorio,
        Equlibrio_local=equil,
        Medicion_local=Medicion,
        int_muestro_local=int_muestro
    )
    Cv_list.append(Cv)
    Emean_list.append(Emean)
    #print(f"β={beta:.3f}  Cv={Cv:.4f}  <E>={Emean:.4f}")

Cv = np.array(Cv_list)
Emean = np.array(Emean_list)

beta_c = 0.5*np.log(1+np.sqrt(2))   
plt.figure(figsize=(6,4))
plt.plot(betas, Cv, lw=2, color='k')
plt.axvline(beta_c, color='r', linestyle='-', alpha=0.7, label='β crítico (teoría)')
plt.xlabel("Thermodynamic β ")
plt.ylabel("Specific Heat from simulation ")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("1.b.pdf", bbox_inches="tight")
plt.close()