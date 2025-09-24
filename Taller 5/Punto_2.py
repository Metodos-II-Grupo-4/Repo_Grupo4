import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
ax[0].plot(t, U, color="green")
ax[1].plot(t, Np, color="blue")
ax[2].plot(t, Pu, color="gray")
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
def sde_rk2(dt, steps, Y0):
    Y = np.array(Y0, dtype=float)
    traj = np.zeros((steps, 3))
    traj[0] = Y
    for i in range(1, steps):
        U, Np, Pu = Y

        # Drift terms
        muU = A - lambda_U * U
        muNp = lambda_U * U - lambda_Np * Np
        muPu = lambda_Np * Np - B * Pu

        # Volatility terms
        sigmaU = np.sqrt(abs(A + lambda_U * U))
        sigmaNp = np.sqrt(abs(lambda_U * U + lambda_Np * Np))
        sigmaPu = np.sqrt(abs(lambda_Np * Np + B * Pu))

        W = np.random.normal()
        S = np.random.choice([-1, 1])

        # K1
        K1U = dt * muU + (W + S) * np.sqrt(dt) * sigmaU
        K1Np = dt * muNp + (W + S) * np.sqrt(dt) * sigmaNp
        K1Pu = dt * muPu + (W + S) * np.sqrt(dt) * sigmaPu

        # K2
        K2U = dt * (A - lambda_U * (U + K1U)) + (W + S) * np.sqrt(dt) * sigmaU
        K2Np = dt * (lambda_U * (U + K1U) - lambda_Np * (Np + K1Np)) + (W + S) * np.sqrt(dt) * sigmaNp
        K2Pu = dt * (lambda_Np * (Np + K1Np) - B * (Pu + K1Pu)) + (W + S) * np.sqrt(dt) * sigmaPu

        # Update
        Y = Y + 0.5 * np.array([K1U + K2U, K1Np + K2Np, K1Pu + K2Pu])
        traj[i] = Y
    return traj

dt = 0.01
steps = int(t_max / dt)
t_sde = np.linspace(0, t_max, steps)

trajectories_sde = [sde_rk2(dt, steps, Y0) for _ in range(5)]

# -------------------------------
# 2.c Simulación exacta (Gillespie)
# -------------------------------
def gillespie(t_max, Y0):
    t = 0
    Y = np.array(Y0, dtype=int)
    times = [t]
    traj = [Y.copy()]
    while t < t_max:
        U, Np, Pu = Y
        rates = np.array([A, lambda_U * U, lambda_Np * Np, B * Pu])
        rate_sum = rates.sum()
        if rate_sum <= 0:
            break

        tau = np.random.exponential(1 / rate_sum)
        r = np.random.choice(len(rates), p=rates / rate_sum)

        # Update state
        if r == 0:
            Y[0] += 1
        elif r == 1:
            Y[0] -= 1; Y[1] += 1
        elif r == 2:
            Y[1] -= 1; Y[2] += 1
        elif r == 3:
            Y[2] -= 1

        t += tau
        times.append(t)
        traj.append(Y.copy())

    return np.array(times), np.array(traj)

gillespie_trajs = [gillespie(t_max, Y0) for _ in range(5)]

# -------------------------------
# Gráficas
# -------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# (a) determinista
axes[0].plot(t_vals, U_det, label="U determinista")
axes[0].plot(t_vals, Np_det, label="Np determinista")
axes[0].plot(t_vals, Pu_det, label="Pu determinista")
axes[0].set_ylabel("Cantidad")
axes[0].set_title("2.a Sistema determinista")
axes[0].legend()

# (b) SDE
axes[1].plot(t_vals, U_det, "k--", alpha=0.5)
axes[1].plot(t_vals, Np_det, "k--", alpha=0.5)
axes[1].plot(t_vals, Pu_det, "k--", alpha=0.5)
for traj in trajectories_sde:
    axes[1].plot(t_sde, traj[:,0], alpha=0.7)
    axes[1].plot(t_sde, traj[:,1], alpha=0.7)
    axes[1].plot(t_sde, traj[:,2], alpha=0.7)
axes[1].set_ylabel("Cantidad")
axes[1].set_title("2.b SDE - 5 trayectorias")

# (c) Gillespie
axes[2].plot(t_vals, U_det, "k--", alpha=0.5)
axes[2].plot(t_vals, Np_det, "k--", alpha=0.5)
axes[2].plot(t_vals, Pu_det, "k--", alpha=0.5)
for times, traj in gillespie_trajs:
    axes[2].step(times, traj[:,0], alpha=0.7, where="post")
    axes[2].step(times, traj[:,1], alpha=0.7, where="post")
    axes[2].step(times, traj[:,2], alpha=0.7, where="post")
axes[2].set_ylabel("Cantidad")
axes[2].set_xlabel("Tiempo (días)")
axes[2].set_title("2.c Gillespie - 5 trayectorias")

plt.tight_layout()
plt.show()
