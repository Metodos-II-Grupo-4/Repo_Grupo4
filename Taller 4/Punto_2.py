# Punto 2 - Tarea 4 Métodos Computacionales II - Partial Differential Equations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pde

# =============================
# Sistema base
# =============================
grid = pde.CartesianGrid([[0, 3], [0, 3]], [200, 200], periodic=False)

# Estado inicial con ruido gaussiano estándar
u = pde.ScalarField.random_normal(grid, 0.5, 0.05)
v = pde.ScalarField.random_normal(grid, 0.25, 0.05)
state = pde.FieldCollection([u, v])

# Constantes
alpha = 0.00028  # Activador
beta = 0.05      # Inhibidor

# Funciones
F = "u - u**3 - v - 0.05"
G = "10 * (u-v)"

# Expresión
expr = {
    "u": f"{alpha}*laplace(u) + {F}",
    "v": f"{beta}*laplace(v) + {G}"
}

# Resolver sistema base
equation = pde.PDE(expr, bc="neumann")
storage = pde.MemoryStorage()
sol = equation.solve(state, t_range=16., dt=1e-4, tracker=["progress", storage.tracker(0.01)])


# =============================
# Ruido aumentado
# =============================
u_r_mas = pde.ScalarField.random_normal(grid, 0.5, 0.2)
v_r_mas = pde.ScalarField.random_normal(grid, 0.25, 0.2)
state_r_mas = pde.FieldCollection([u_r_mas, v_r_mas])

expr_r_mas = {
    "u": f"{alpha}*laplace(u) + {F}",
    "v": f"{beta}*laplace(v) + {G}"
}

equation_r_mas = pde.PDE(expr_r_mas, bc="neumann")
storage_r_mas = pde.MemoryStorage()
sol_r_mas = equation_r_mas.solve(state_r_mas, t_range=16., dt=1e-4, tracker=["progress", storage_r_mas.tracker(0.01)])


# =============================
# Ruido disminuido
# =============================
u_r_menos = pde.ScalarField.random_normal(grid, 0.5, 0.01)
v_r_menos = pde.ScalarField.random_normal(grid, 0.25, 0.01)
state_r_menos = pde.FieldCollection([u_r_menos, v_r_menos])

expr_r_menos = {
    "u": f"{alpha}*laplace(u) + {F}",
    "v": f"{beta}*laplace(v) + {G}"
}

equation_r_menos = pde.PDE(expr_r_menos, bc="neumann")
storage_r_menos = pde.MemoryStorage()
sol_r_menos = equation_r_menos.solve(state_r_menos, t_range=16., dt=1e-4, tracker=["progress", storage_r_menos.tracker(0.01)])


# =============================
# Activación x4
# =============================
alpha_ax4 = 0.00112
expr_ax4 = {
    "u": f"{alpha_ax4}*laplace(u) + {F}",
    "v": f"{beta}*laplace(v) + {G}"
}

equation_ax4 = pde.PDE(expr_ax4, bc="neumann")
storage_ax4 = pde.MemoryStorage()
sol_ax4 = equation_ax4.solve(state, t_range=16., dt=1e-4, tracker=["progress", storage_ax4.tracker(0.01)])


# =============================
# Activación /4
# =============================
alpha_a4 = 0.00007
expr_a4 = {
    "u": f"{alpha_a4}*laplace(u) + {F}",
    "v": f"{beta}*laplace(v) + {G}"
}

equation_a4 = pde.PDE(expr_a4, bc="neumann")
storage_a4 = pde.MemoryStorage()
sol_a4 = equation_a4.solve(state, t_range=16., dt=1e-4, tracker=["progress", storage_a4.tracker(0.01)])


# =============================
# Modelo Schnakenberg (puntos)
# =============================
grid_puntos = pde.CartesianGrid([[0, 3], [0, 3]], [200, 200], periodic=True)
u_puntos = pde.ScalarField.random_normal(grid_puntos, 0.5, 0.05)
v_puntos = pde.ScalarField.random_normal(grid_puntos, 0.25, 0.05)
state_puntos = pde.FieldCollection([u_puntos, v_puntos])

a = 0.025
b = 1.55
alpha_puntos = 0.0005
dx = 0.1
dy = 0.1

F_puntos = f"{a} + v*u**2 - u"
G_puntos = f"{b} - v*u**2"

expr_puntos = {
    "u": f"{alpha_puntos}*laplace(u) + {F_puntos}",
    "v": f"{dx}*d2_dx2(v) + {dy}*d2_dy2(v) + {G_puntos}"
}

equation_puntos = pde.PDE(expr_puntos)
storage_puntos = pde.MemoryStorage()
sol_puntos = equation_puntos.solve(state_puntos, t_range=30., dt=1e-4, tracker=["progress", storage_puntos.tracker(0.01)])


# =============================
# Modelo Schnakenberg (rayas)
# =============================
v_rayas = pde.ScalarField.random_normal(grid_puntos, 0.4, 0.05)
state_rayas = pde.FieldCollection([u_puntos, v_rayas])

a = 0.025
b = 1.55
alpha_rayas = 0.000125
dx = 0.001
dy = 0.005

F_rayas = f"{a} + v*u**2 - u"
G_rayas = f"{b} - v*u**2"

expr_rayas = {
    "u": f"{alpha_rayas}*laplace(u) + {F_rayas}",
    "v": f"{dx}*d2_dx2(v) + {dy}*d2_dy2(v) + {G_rayas}"
}

equation_rayas = pde.PDE(expr_rayas)
storage_rayas = pde.MemoryStorage()
sol_rayas = equation_rayas.solve(state_rayas, t_range=30., dt=1e-4, tracker=["progress", storage_rayas.tracker(0.01)])


# GRAFICAMOS RUIDO

# Filas = u o v. Columnas = valor de ruido
fig_ruido, axes_ruido = plt.subplots(2, 3, figsize=(15, 5))

# Caso 1: ruido disminuido
im0 = axes_ruido[0][0].imshow(sol_r_menos.data[0], origin='lower', cmap='viridis')
axes_ruido[0][0].set_title("Ruido 0.01 (u)")
fig_ruido.colorbar(im0, ax=axes_ruido[0][0], fraction=0.046, pad=0.04)

im1 = axes_ruido[1][0].imshow(sol_r_menos.data[1], origin='lower', cmap='viridis')
axes_ruido[1][0].set_title("Ruido 0.01 (v)")
fig_ruido.colorbar(im1, ax=axes_ruido[1][0], fraction=0.046, pad=0.04)

# Caso 2: ruido normal
im2 = axes_ruido[0][1].imshow(sol.data[0], origin='lower', cmap='viridis')
axes_ruido[0][1].set_title("Ruido 0.05 (u)")
fig_ruido.colorbar(im2, ax=axes_ruido[0][1], fraction=0.046, pad=0.04)

im3 = axes_ruido[1][1].imshow(sol.data[1], origin='lower', cmap='viridis')
axes_ruido[1][1].set_title("Ruido 0.05 (v)")
fig_ruido.colorbar(im3, ax=axes_ruido[1][1], fraction=0.046, pad=0.04)

# Caso 3: ruido aumentado
im4 = axes_ruido[0][2].imshow(sol_r_mas.data[0], origin='lower', cmap='viridis')
axes_ruido[0][2].set_title("Ruido 0.20 (u)")
fig_ruido.colorbar(im4, ax=axes_ruido[0][2], fraction=0.046, pad=0.04)

im5 = axes_ruido[1][2].imshow(sol_r_mas.data[1], origin='lower', cmap='viridis')
axes_ruido[1][2].set_title("Ruido 0.20 (v)")
fig_ruido.colorbar(im5, ax=axes_ruido[1][2], fraction=0.046, pad=0.04)

plt.suptitle("Comparación de patrón de escamas Pez Globo para diferentes niveles de ruido", fontsize=14)
plt.tight_layout()
plt.savefig("2_Mas ruido, menor cantidad de grupos aislados.png")

# Filas = u o v. Columnas = valor de activación
fig_alpha, axes_alpha = plt.subplots(2, 3, figsize=(15, 5))

# Caso 1: activación disminuida
im6 = axes_alpha[0][0].imshow(sol_a4.data[0], origin='lower', cmap='viridis')
axes_alpha[0][0].set_title("α= 0.00007 (u)")
fig_alpha.colorbar(im6, ax=axes_alpha[0][0], fraction=0.046, pad=0.04)

im7 = axes_alpha[1][0].imshow(sol_a4.data[1], origin='lower', cmap='viridis')
axes_alpha[1][0].set_title("α= 0.00007 (v)")
fig_alpha.colorbar(im7, ax=axes_alpha[1][0], fraction=0.046, pad=0.04)

# Caso 2: activación normal
im8 = axes_alpha[0][1].imshow(sol.data[0], origin='lower', cmap='viridis')
axes_alpha[0][1].set_title("α= 0.00028 (u)")
fig_alpha.colorbar(im8, ax=axes_alpha[0][1], fraction=0.046, pad=0.04)

im9 = axes_alpha[1][1].imshow(sol.data[1], origin='lower', cmap='viridis')
axes_alpha[1][1].set_title("α= 0.00028 (v)")
fig_alpha.colorbar(im9, ax=axes_alpha[1][1], fraction=0.046, pad=0.04)

# Caso 3: activación aumentada
im10 = axes_alpha[0][2].imshow(sol_ax4.data[0], origin='lower', cmap='viridis')
axes_alpha[0][2].set_title("α= 0.00112 (u)")
fig_alpha.colorbar(im10, ax=axes_alpha[0][2], fraction=0.046, pad=0.04)

im11 = axes_alpha[1][2].imshow(sol_ax4.data[1], origin='lower', cmap='viridis')
axes_alpha[1][2].set_title("α= 0.00112 (v)")
fig_alpha.colorbar(im11, ax=axes_alpha[1][2], fraction=0.046, pad=0.04)

plt.suptitle("Comparación de patrón de escamas Pez Globo para diferentes activaciones", fontsize=14)
plt.tight_layout()
plt.savefig("2_Menor activacion, mayor resolucion.png")

# Filas = u o v. Puntos del Cheetah
fig_puntos, axes_puntos = plt.subplots(2, 1, figsize=(15, 5))

im12 = axes_puntos[0].imshow(sol_puntos.data[0], origin='lower', cmap='inferno')
axes_puntos[0].set_title("Puntos Cheetah (u)")
fig_puntos.colorbar(im12, ax=axes_puntos[0], fraction=0.046, pad=0.04)

im13 = axes_puntos[1].imshow(sol_puntos.data[1], origin='lower', cmap='inferno')
axes_puntos[1].set_title("Puntos Cheetah (v)")
fig_puntos.colorbar(im13, ax=axes_puntos[1], fraction=0.046, pad=0.04)

plt.suptitle("Puntos en el pelaje del Cheetah", fontsize=14)
plt.tight_layout()
plt.savefig("2_Cambio en F y G, nuevo patron (puntos felinos).png")

# Filas = u o v. Rayas de la cebra
fig_rayas, axes_rayas = plt.subplots(2, 1, figsize=(15, 5))

im14 = axes_rayas[0].imshow(sol_rayas.data[0], origin='lower', cmap='gray')
axes_rayas[0].set_title("Rayas (u)")
fig_rayas.colorbar(im14, ax=axes_rayas[0], fraction=0.046, pad=0.04)

im15 = axes_rayas[1].imshow(sol_rayas.data[1], origin='lower', cmap='gray')
axes_rayas[1].set_title("Rayas (v)")
fig_rayas.colorbar(im15, ax=axes_rayas[1], fraction=0.046, pad=0.04)

plt.suptitle("Rayas en el pelaje de la cebra", fontsize=14)
plt.tight_layout()
plt.savefig("2_Cambio en F y G + constantes, nuevo patron (rayas de cebra).png")