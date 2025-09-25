import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def init_spins(N, seed=None):
    aleatorio = np.random.default_rng(seed)
    spins = aleatorio.choice([-1, 1], size=(N, N))
    return spins, aleatorio

def Vecinos_cercanos_sum(spins, i, j):
    N = spins.shape[0]
    return (spins[(i+1) % N, j] + spins[(i-1) % N, j] +
            spins[i, (j+1) % N] + spins[i, (j-1) % N])

def Energia_local(spins, J):
    # epsilon_0 = sum_{i,j} ε_ij con 4 vecinos (cuenta cada enlace dos veces)
    up    = np.roll(spins, -1, axis=0)
    down  = np.roll(spins,  1, axis=0)
    left  = np.roll(spins,  1, axis=1)
    right = np.roll(spins, -1, axis=1)
    epsilon = -J * spins * (up + down + left + right)
    return epsilon.sum()

def Metropolis(spins, beta, J, aleatorio):
    N = spins.shape[0]
    i = aleatorio.integers(0, N)
    j = aleatorio.integers(0, N)
    s = spins[i, j]
    Snn = Vecinos_cercanos_sum(spins, i, j)

    dH  = 2.0 * J * s * Snn          # para aceptación
    dEp = 4.0 * J * s * Snn          # para actualizar E(t) de tus notas
    dm  = -2 * s                     # cambio de magnetización

    accepted = False
    if dH <= 0.0 or aleatorio.random() < np.exp(-beta * dH):
        spins[i, j] = -s
        accepted = True
        return True, dEp, dm
    else:
        return False, 0.0, 0.0

# Parámetros
N = 50
kB = 1
J = 1
beta = 1/2
epocas = 100
norm_M = 'per_spin'
seed = 42

# Estado inicial y normalización 
spins, aleatorio = init_spins(N, seed)
eps_total  = Energia_local(spins, J) #suma de energías locales
mag_sum    = spins.sum()              

E = np.empty(epocas + 1, dtype=float)
M = np.empty(epocas + 1, dtype=float)

E[0] = eps_total / (4.0 * N * N)
M[0] = (mag_sum / (N * N)) if norm_M == 'per_spin' else (mag_sum / N)

# Simulación por épocas 
intentos_por_epoca = N * N
accepts = 0

# Épocas para snapshots tipo "Antes/Durante/Después"
snapshot_epochs = {0, 100, epocas}
snaps = {}
if 0 in snapshot_epochs:
    snaps[0] = spins.copy()

for e in range(1, epocas + 1):
    for _ in range(intentos_por_epoca):
        ok, dEp, dm = Metropolis(spins, beta, J, aleatorio)
        if ok:
            accepts  += 1
            eps_total += dEp
            mag_sum   += dm

    # fin de la época 
    E[e] = eps_total / (4.0 * N * N)
    M[e] = (mag_sum / (N * N)) if norm_M == 'per_spin' else (mag_sum / N)
    if e in snapshot_epochs:
        snaps[e] = spins.copy()

acc_rate = accepts / (epocas * intentos_por_epoca)

ep = np.arange(len(E))

fig = plt.figure(figsize=(12,4))

# Recuadro del medio: Evolución de E y M
ax = plt.subplot(1,3,2)
ax.plot(ep, E, lw=2, color="black", label="Energía")
ax.plot(ep, M, lw=2, color="red",   label="Magnetización")
ax.set_xlabel("Épocas")
ax.set_ylim(-1.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend()

#Primer recuadro: “Antes”
axL = plt.subplot(1,3,1)
axL.imshow(snaps[min(snaps.keys())], cmap="coolwarm", vmin=-1, vmax=1)
axL.set_title("Antes")
axL.axis("off")

#Tercer recuadro: “Después”
axR = plt.subplot(1,3,3)
axR.imshow(snaps[max(snaps.keys())], cmap="coolwarm", vmin=-1, vmax=1)
axR.set_title("Después")
axR.axis("off")

plt.tight_layout()
plt.savefig("1.a.pdf", bbox_inches="tight")
plt.close(fig)

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

tamano_promedio_dominios(spins)

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