import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from joblib import Parallel, delayed
# Parámetros
A = 1000
B = 20
t_half_U = 23.4 / 1440   # pasamos de minutos a días
t_half_Np = 2.36
lambda_U = np.log(2) / t_half_U
lambda_Np = np.log(2) / t_half_Np

# -------------------------------
# 2.a Sistema determinista
# -------------------------------
t_max = 30
tol = 1e-2
Y0 = [10, 10, 10]  # U, Np, Pu

def odes(t, Y):
    U, Np, Pu = Y # Vector de estado Y
    return [A - lambda_U * U, 
            lambda_U * U - lambda_Np * Np,
            lambda_Np * Np - B * Pu] # Derivada de Y

def derivada_nula(t, Y, *args):
    return np.linalg.norm(odes(t, Y)) - tol

derivada_nula.terminal = False
derivada_nula.direction = -1

fig, ax = plt.subplots(1,3, figsize=(12,4))
t = np.linspace(0, 30, 10000)
for a in ax:
    a.set_xlabel("t(días)")
ax[0].set_ylabel("U(N)")
ax[1].set_ylabel("Np(N)")
ax[2].set_ylabel("Pu(N)")

solucion = solve_ivp(fun=odes,
                       t_span=(0, 30),
                       y0=np.array([10., 10., 10.]),
                       max_step=0.003,
                       events=derivada_nula,
                       dense_output=True,
                       method="RK45")

t = solucion.t
U, Np, Pu = solucion.y
ax[0].set_xscale("log")
ax[0].plot(t, U, color="green", label="U")
ax[1].plot(t, Np, color="blue", label="Np")
ax[2].plot(t, Pu, color="gray", label="Pu")

# Verificar si hubo steady state
if len(solucion.y_events[0]) > 0:
    steady_state = solucion.y_events[0][0]
    U_event, Np_event, Pu_event = steady_state
    t_event = solucion.t_events[0][0]
    ax[0].scatter(t_event, U_event, color="red", label=f"Steady state (t={t_event:.2f}, U={U_event:.2f})")
    ax[1].scatter(t_event, Np_event, color="red", label=f"Steady state (t={t_event:.2f}, Np={Np_event:.2f})")
    ax[2].scatter(t_event, Pu_event, color="red", label=f"Steady state (t={t_event:.2f}, Pu={Pu_event:.2f})")
    ax[0].hlines(U_event, xmin=0, xmax=30, color="k")
    ax[1].hlines(Np_event, xmin=0, xmax=30, color="k")
    ax[2].hlines(Pu_event, xmin=0, xmax=30, color="k")

else:
    ax[0].plot([], [], ' ', label=f"NO STEADY STATE (tol={tol})")
    ax[1].plot([], [], ' ', label=f"NO STEADY STATE (tol={tol})")
    ax[2].plot([], [], ' ', label=f"NO STEADY STATE (tol={tol})")

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[0].set_title("Uranio en función de t")
ax[1].set_title("Neptunio en función de t")
ax[2].set_title("Plutonio en función de t")
plt.tight_layout()
plt.savefig("2.a.pdf")

# -------------------------------
# 2.b Ecuación diferencial estocástica
# -------------------------------
def SDE_RK2(dt, steps, Y0): #Definimos nuestra función de Runge-Kutta estocástico orden 2.
    traj = np.zeros((steps, 3)) #Acá guardaremos la solución 
    traj[0] = Y0 #Ponemos la condición inicial en la solución
    Y = np.array(Y0, dtype=float) #Creamos el vector que se actualizará cada iteración
    for i in tqdm(range(1, steps)):
        U, Np, Pu = Y
        
        #De la definición de W y S, elegidos aleatoriamente cada iteración
        W = np.random.normal()
        S = np.random.choice([-1, 1])

        #Drift
        muU = A - lambda_U * U
        muNp = lambda_U * U - lambda_Np * Np
        muPu = lambda_Np * Np - B * Pu

        #Volatilidad
        sigmaU = np.sqrt(A + lambda_U * U)
        sigmaNp = np.sqrt(lambda_U * U + lambda_Np * Np)
        sigmaPu = np.sqrt(lambda_Np * Np + B * Pu)

        #K1
        K1U = dt * muU + (W - S) * np.sqrt(dt) * sigmaU
        K1Np = dt * muNp + (W - S) * np.sqrt(dt) * sigmaNp
        K1Pu = dt * muPu + (W - S) * np.sqrt(dt) * sigmaPu

        #K2
        K2U = dt * (A - lambda_U * (U + K1U)) + (W + S) * np.sqrt(dt) * sigmaU
        K2Np = dt * (lambda_U * (U + K1U) - lambda_Np * (Np + K1Np)) + (W + S) * np.sqrt(dt) * sigmaNp
        K2Pu = dt * (lambda_Np * (Np + K1Np) - B * (Pu + K1Pu)) + (W + S) * np.sqrt(dt) * sigmaPu

        #Actualizamos el vector y añadimos a la trayectoria del sistema.
        Y = Y + 0.5 * np.array([K1U + K2U, K1Np + K2Np, K1Pu + K2Pu])
        traj[i] = Y
    return traj #Retornamos la trayectoria

dt = 0.001
steps = int(t_max / dt)
t_sde = np.linspace(0, t_max, steps)
trajectories_sde = [SDE_RK2(dt, steps, Y0) for _ in range(5)] #5 trayectorias

#Primero, graficamos la solución anterior para comparar.
fig, ax = plt.subplots(1,3, figsize=(12,4))
for a in ax:
    a.set_xlabel("t(días)")
ax[0].set_ylabel("U(N)")
ax[1].set_ylabel("Np(N)")
ax[2].set_ylabel("Pu(N)")
ax[0].plot(t, U, color="k", label="U (a)", linestyle="--", zorder=5)
ax[1].plot(t, Np, color="k", label="Np (a)", linestyle="--", zorder=5)
ax[2].plot(t, Pu, color="k", label="Pu (a)", linestyle="--", zorder=5)
#Ahora si, sobre cada subplot se grafican las 5 trayectorias.
for traj in trajectories_sde:
    ax[0].plot(t_sde, traj[:,0], alpha=0.7)
    ax[1].plot(t_sde, traj[:,1], alpha=0.7)
    ax[2].plot(t_sde, traj[:,2], alpha=0.7)
ax[0].set_title("Uranio en función de t")
ax[1].set_title("Neptunio en función de t")
ax[2].set_title("Plutonio en función de t")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.tight_layout()
plt.savefig("2.b.pdf")


# -------------------------------
# 2.c Simulación exacta (Gillespie)
# -------------------------------
def gillespie(t_max, Y0):
    traj = [np.array(Y0, dtype=float)] #Empezamos igual que en el anterior modelo estocástico
    Y = np.array(Y0, dtype=float)
    #Creamos también array de tiempos
    tiempos = [0]
    t = 0
    while t < t_max:
        U, Np, Pu = Y
        rates = np.array([A, lambda_U * U, lambda_Np * Np, B * Pu]) #Definimos las tasas como en el documento
        rate_sum = rates.sum()
        if rate_sum <= 0: #Condición trivial, las probabilidades son positivas.
            break

        tau = np.random.exponential(1 / rate_sum) #Tiempo de siguiente reacción
        r = np.random.choice(len(rates), p=rates / rate_sum) #Elegimos la siguiente reacción

        # Recalculamos las tasas (como se definió en el documento)
        if r == 0:
            Y[0] += 1
        elif r == 1:
            Y[0] -= 1
            Y[1] += 1
        elif r == 2:
            Y[1] -= 1
            Y[2] += 1
        elif r == 3:
            Y[2] -= 1

        t += tau
        tiempos.append(t)
        traj.append(Y.copy())

    return np.array(tiempos), np.array(traj)

gillespie_trajs = [gillespie(t_max, Y0) for _ in range(5)] #Lo mismo que antes.

#Primero, graficamos la solución anterior para comparar.
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a in ax:
    a.set_xlabel("t(días)")
    
ax[0].set_ylabel("U(N)")
ax[1].set_ylabel("Np(N)")
ax[2].set_ylabel("Pu(N)")

# Graficar la solución anterior para comparar
ax[0].plot(t, U, color="k", label="U (a)", linestyle="--", zorder=5)
ax[1].plot(t, Np, color="k", label="Np (a)", linestyle="--", zorder=5)
ax[2].plot(t, Pu, color="k", label="Pu (a)", linestyle="--", zorder=5)

# Ahora, sobre cada subplot se grafican las 5 trayectorias
for tiempos_traj, especies_traj in gillespie_trajs:
    ax[0].plot(tiempos_traj, especies_traj[:, 0], alpha=0.7)
    ax[1].plot(tiempos_traj, especies_traj[:, 1], alpha=0.7)
    ax[2].plot(tiempos_traj, especies_traj[:, 2], alpha=0.7)

ax[0].set_title("Uranio en función de t")
ax[1].set_title("Neptunio en función de t")
ax[2].set_title("Plutonio en función de t")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.tight_layout()
plt.savefig("2.c.pdf")

# -------------------------------
# 2.d. Estimación de probabilidad para el plutonio
# -------------------------------
def run_RK(Y0, steps, dt, A, lambda_U, lambda_Np, B):
    for _ in range(1, steps):
        U, Np, Pu = Y0
        
        #De la definición de W y S, elegidos aleatoriamente cada iteración
        W = np.random.normal()
        S = np.random.choice([-1, 1])

        #Drift
        muU = A - lambda_U * U
        muNp = lambda_U * U - lambda_Np * Np
        muPu = lambda_Np * Np - B * Pu

        #Volatilidad
        sigmaU = np.sqrt(A + lambda_U * U)
        sigmaNp = np.sqrt(lambda_U * U + lambda_Np * Np)
        sigmaPu = np.sqrt(lambda_Np * Np + B * Pu)

        #K1
        K1U = dt * muU + (W - S) * np.sqrt(dt) * sigmaU
        K1Np = dt * muNp + (W - S) * np.sqrt(dt) * sigmaNp
        K1Pu = dt * muPu + (W - S) * np.sqrt(dt) * sigmaPu

        #K2
        K2U = dt * (A - lambda_U * (U + K1U)) + (W + S) * np.sqrt(dt) * sigmaU
        K2Np = dt * (lambda_U * (U + K1U) - lambda_Np * (Np + K1Np)) + (W + S) * np.sqrt(dt) * sigmaNp
        K2Pu = dt * (lambda_Np * (Np + K1Np) - B * (Pu + K1Pu)) + (W + S) * np.sqrt(dt) * sigmaPu

        #Actualizamos el vector y añadimos a la trayectoria del sistema.
        Y0 = Y0 + 0.5 * np.array([K1U + K2U, K1Np + K2Np, K1Pu + K2Pu])

        if Pu >= 80:
            return 1  #Superó 80
    return 0  #No superó

def run_gillespie(Y0, t_max, A, lambda_U, lambda_Np, B):
    U, Np, Pu = Y0
    t = 0.0
    while t < t_max:
        #Otra vez, definimos los rates como en el documento
        r0 = A
        r1 = lambda_U * U
        r2 = lambda_Np * Np
        r3 = B * Pu
        rate_sum = r0 + r1 + r2 + r3
        if rate_sum <= 0:
            break

        tau = np.random.exponential(1.0 / rate_sum)
        r = np.random.random() * rate_sum

        if r < r0:
            U += 1
        elif r < r0 + r1:
            U -= 1
            Np += 1
        elif r < r0 + r1 + r2:
            Np -= 1
            Pu += 1
        else:
            Pu -= 1

        if Pu >= 80:
            return 1
        t += tau
    return 0

def estimate_probs(Y0, steps, t_max, dt, A, lambda_U, lambda_Np, B, n_iter=1000, n_jobs=-1):
    # Run RK secuencial (loop normal)
    rk_results = Parallel(n_jobs=n_jobs)(
        delayed(run_RK)(Y0, steps, dt, A, lambda_U, lambda_Np, B) for _ in range(n_iter)
    )
    
    # Run Gillespie en paralelo
    g_results = Parallel(n_jobs=n_jobs)(
        delayed(run_gillespie)(Y0, t_max, A, lambda_U, lambda_Np, B) for _ in range(n_iter)
    )

    prob_RK = np.mean(rk_results)
    prob_gillespie = np.mean(g_results)
    return prob_RK, prob_gillespie


Y0 = np.array([10.0, 10.0, 10.0])
steps = 30000
t_max = 30.0
dt = t_max/steps
prob_RK, prob_gillespie = estimate_probs(Y0, steps, t_max, dt, A, lambda_U, lambda_Np, B)
info = f"La probabilidad por Runge-Kutta orden 2 es ({prob_RK*100}+-{np.sqrt(prob_RK*(1-prob_RK)/1000)*100})%" + "\n" + f"La probabilidad por Gillespie es ({prob_gillespie*100}+-{100*np.sqrt(prob_gillespie*(1-prob_gillespie)/1000)})%"
info += "\n"
info += "Con un dt de 0.001 (lo mostrado arriba) o de 0.003 (P=(15.3+-1.1)%), la probabilidad con RK queda en un rango muy parecido a la de Gillespie. La probabilidad de Gillespie es estable, pues solo depende de t_max. \n"
info += "Sin embargo, la probabilidad de RK cambia mucho de acuerdo al dt usado. Con dt=0.01, se consiguio una probabilidad  (11.6+-1.1)%. Haciendo dt = 0.03, se consiguio una probabilidad (5.3+-0.7)%. \n"
info += "Es decir, en el limite dt--->0, Range Kutta debe converger a una probabilidad considerablemente mayor que Gillespie. Sin embargo, no se hizo la simulacion en este script porque demoraria una eternidad. \n"
info += "De hecho, correr las simulaciones RK con el menor dt aca requirio de 12 minutos. Habria que simular por más si se quiere ver otra probabilidad  para RK en otro orden de magnitud."
info += "Ese dt menor fue dt=0.0003, para el cual se obtuvo una P(RK) = (29.9+-1.4)%"
info += "En conclusion, la probabilidad P(Pu >= 80) es estable para el metodo estocastico de Gillespie, mas no para el de SRK2."
np.savetxt("2.d.txt", [info], fmt="%s")