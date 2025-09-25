import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib import rcParams
import imageio_ffmpeg

L, N, c = 64.0, 512, 2.5
L*=2
dx = L / N
x = np.linspace(0.0, L, N, endpoint=False)

def soliton1(x_, c_, L_):
    return 0.5 * (1 / np.cosh(0.5 * c_ * (x_ - L_/2)))**2

u0 = soliton1(x, c, L)
def dudx(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
def dudx3(u, dx):
    return (np.roll(u, -2) - 2*np.roll(u, -1) + 2*np.roll(u, 1) - np.roll(u, 2)) / (2.0 * dx**3)
def F(u, dx):
    return -6.0 * u * dudx(u, dx) - dudx3(u, dx)
def rk4_step(u, dt, dx): #Runge kutta de 4 orden para korteweg deVries
    k1 = dt * F(u, dx)
    k2 = dt * F(u + 0.5*k1, dx)
    k3 = dt * F(u + 0.5*k2, dx)
    k4 = dt * F(u + k3, dx)
    return u + (k1 + 2*k2 + 2*k3 + k4) / 6.0

dt = 0.1 * dx**3         
T  = 11.0                
n_steps = int(T / dt)

# Submuestreo para animación
max_frames = 300
sample_every = max(1, (n_steps + 1) // max_frames) 
sx           = max(1, N // 600)


u = u0.copy()
frames = [u.copy()]
for i in range(1, n_steps + 1):
    u = rk4_step(u, dt, dx)
    if i % sample_every == 0:
        frames.append(u.copy())

U = np.array(frames)     

x_d = x[::sx]
U_d = U[:, ::sx]
tvals = np.arange(U_d.shape[0]) * (dt * sample_every)

ymin, ymax = np.quantile(U_d, [0.01, 0.99]) # Ajusta automaticamente la grafica
pad = 0.1 * (ymax - ymin + 1e-12)

fig, ax = plt.subplots(figsize=(10, 5))
(line,) = ax.plot(x_d, U_d[0], lw=2)
title = ax.set_title('Evolución KdV — t = 0.000')
ax.set_xlim(0.0, L)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
plt.tight_layout()

def init():
    line.set_ydata(U_d[0])
    title.set_text('Evolución KdV — t = 0.000')
    return line, title

def update(i):
    line.set_ydata(U_d[i]) #Actualiza nuestra animacion con los datos de u y de t
    title.set_text(f'Evolución KdV — t = {tvals[i]:.3f}')
    return line, title

fps = 20 
anim = animation.FuncAnimation(
    fig, update, init_func=init,
    frames=U_d.shape[0],
    interval=1000/fps, 
    blit=True
)

out = 'kdvsimulation1.mp4'
rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
anim.save(out, writer='ffmpeg', fps=fps)
print("Guardado en:", os.path.abspath(out))
plt.close(fig)
