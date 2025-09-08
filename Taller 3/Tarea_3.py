import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

#PUNTO 1
#1.a. Sistema depredador-presa
def F_1_a(t, Y): #Recibe tiempo de simulación t y vector de estado Y
    x, y = Y #Son dos ecuaciones de primer orden, solo se necesitan 2 entradas en el vector de estado
    return np.array([2*x - 1.5*x*y,
                     -0.3*y + 0.4*x*y]) #Retorna la derivada temporal del vector de estado
    
solucion_1_a = solve_ivp(
    fun=F_1_a, #Pasamos la función de vector de estado
    t_span=(0,50.), #Tiempo de simulación 50s
    y0=np.array([3., 2.]), #Valores iniciales x_0 = 3, y_0 = 2
    max_step=0.05, #dt=0.05
    dense_output=True,
    method="DOP853" #Usamos el método  Runge-Kutta implícito de la familia de orden 5 RadauIIA
)

def V(x,y): #Definimos la función que se debe conservar
    return 0.4*x - 0.3*np.log(x) + 1.5*y - 2*np.log(y)

t = np.linspace(0, 50, 200) #Creamos el linspace del tiempo en el que se va a evaluar la solucón
x = solucion_1_a.sol(t)[0] #Evaluamos la solución para x en t
y = solucion_1_a.sol(t)[1] #Evaluamos la solución para y en t

fig, ax = plt.subplots(3, 1, figsize=(10,20))
ax[0].plot(t, x) #Graficamos x vs. t
ax[0].set_title("x(t) vs. t")
ax[0].set_xlabel("t")
ax[0].set_ylabel("x(t)")
ax[1].plot(t, y)
ax[1].set_title("y(t) vs. t") #Graficamos y vs. t
ax[1].set_xlabel("t")
ax[1].set_ylabel("y(t)")
ax[2].plot(t, V(x,y)) #Graficamos la cantidad conservada vs. t
ax[2].set_title("V(x,y) vs. t")
ax[2].set_xlabel("t")
ax[2].set_ylabel("V(x,y)")
plt.savefig("1.a.pdf")

#1.b. Problema de Landau
c = 1
q = 7.5284
B_0 = 0.438
E_0 = 0.7423
m = 3.8428
k = 1.0014
def F_1_b(t, Y): #Recibe tiempo de simulación t y vector de estado Y
    x, y, v_x, v_y = Y #Son dos ecuaciones de segundo orden, se necesitan 4 entradas en el vector de estado
    return np.array([v_x,
                     v_y,
                     (q/m)*(E_0*(np.sin(k*x) + k*x*np.cos(k*x)) - (B_0/c)*v_y),
                     v_x*(q*B_0)/(m*c)]) #Retorna la derivada temporal del vector de estado
    
solucion_1_b = solve_ivp(
    fun=F_1_b, #Pasamos la función de vector de estado
    t_span=(0,30.), #Tiempo de simulación 30s
    y0=np.array([0., 0., 0.2, 0.]), #Valores iniciales x_0 = 0, y_0 = 0, vx_0 = 0.2 y vy_0 = 0
    max_step=0.001, #dt=0.03
    dense_output=True,
    atol=1e-9, rtol=1e-9,
    method="DOP853" #Usamos el método explícito Runge Kutta de orden 8
)

def Pi_y(x, y, v_x, v_y): #Definimos el momento conjugado a conservar
    return m*v_y - (q*B_0/c)*x

def E(x, y, v_x, v_y): #Definimos la energía total a conservar
    return (m/2)*(v_x**2 + v_y**2) - q*E_0*x*np.sin(k*x)

t = np.linspace(0, 30, 10000) #Creamos el linspace del tiempo en el que se va a evaluar la solucón
x = solucion_1_b.sol(t)[0] #Evaluamos la solución para x en t
y = solucion_1_b.sol(t)[1] #Evaluamos la solución para y en t
v_x = solucion_1_b.sol(t)[2] #Evaluamos la solución para v_x en t
v_y = solucion_1_b.sol(t)[3] #Evaluamos la solución para v_y en t

fig, ax = plt.subplots(2, 2, figsize=(10,20))
ax[0][0].plot(t, x) #Graficamos x vs. t
ax[0][0].set_title("x(t) vs. t")
ax[0][0].set_xlabel("t")
ax[0][0].set_ylabel("x(t)")
ax[0][1].plot(t, y) #Graficamos y vs. t
ax[0][1].set_title("y(t) vs. t")
ax[0][1].set_xlabel("t")
ax[0][1].set_ylabel("y(t)")
ax[1][0].plot(t, Pi_y(x,y, v_x, v_y)) #Graficamos el momento conjugado vs. t
ax[1][0].set_title("Pi_y(x,y) vs. t")
ax[1][0].set_xlabel("t")
ax[1][0].set_ylabel("Pi_y(x,y)")
ax[1][1].plot(t, E(x,y, v_x, v_y)) #Graficamos la energía total vs. t
ax[1][1].set_title("E(x,y) vs. t")
ax[1][1].set_xlabel("t")
ax[1][1].set_ylabel("E(x,y)")
plt.savefig("1.b.pdf")

#1.c.
G = 1.0
m = 1.7

def F_1_c(t, Y):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = Y #Son cuatro ecuaciones de segundo orden, se necesitan 8 entradas en el vector de estado.
    return np.array([vx1,
                     vy1,
                     vx2,
                     vy2,
                     G*m*(x2-x1)/((x2-x1)**2 + (y2-y1)**2)**1.5,
                     G*m*(y2-y1)/((x2-x1)**2 + (y2-y1)**2)**1.5,
                    -G*m*(x2-x1)/((x2-x1)**2 + (y2-y1)**2)**1.5,
                    -G*m*(y2-y1)/((x2-x1)**2 + (y2-y1)**2)**1.5])

solucion_1_c = solve_ivp(
    fun=F_1_c, #Pasamos la función de vector de estado
    t_span=(0,10.), #Tiempo de simulación 10s
    y0=np.array([0., 0., 1., 1., 0., 0.5, 0., -0.5]), #Valores iniciales (x_1, y_1) = (0, 0), (x_2, y_2) = (1, 1), (vx_1, vy_1) = (0, 0.5), (vx_2, vy_2) = (0, -0.5).
    max_step=0.001, #dt=0.001
    dense_output=True,
    atol=1e-9, rtol=1e-9,
    method="DOP853" #Usamos el método explícito Runge Kutta de orden 8
)

def E_tot(x1, y1, x2, y2, vx1, vy1, vx2, vy2): #E = T + V a conservar
    return (m/2)*(vx1**2 + vy1**2) + (m/2)*(vx2**2 + vy2**2) - G*m/np.sqrt((x2-x1)**2 + (y2-y1)**2)

def L(x1, y1, x2, y2, vx1, vy1, vx2, vy2): #L = r x p a conservar
    return m*(x1*vy1 - y1*vx1) + m*(x2*vy2 - y2*vx2)

t = np.linspace(0, 10, 10000)
x1, y1, x2, y2, vx1, vy1, vx2, vy2 = solucion_1_c.sol(t)

# Cantidades conservadas
E_vals = E_tot(x1, y1, x2, y2, vx1, vy1, vx2, vy2)
L_vals = L(x1, y1, x2, y2, vx1, vy1, vx2, vy2)

# Hacemos subplots
fig, ax = plt.subplots(3, 2, figsize=(12, 10))

ax[0,0].plot(t, x1, label="x1")
ax[0,0].plot(t, x2, label="x2")
ax[0,0].set_title("x1(t), x2(t)")
ax[0,0].legend()

ax[0,1].plot(t, y1, label="y1")
ax[0,1].plot(t, y2, label="y2")
ax[0,1].set_title("y1(t), y2(t)")
ax[0,1].legend()

ax[1,0].plot(t, vx1, label="vx1")
ax[1,0].plot(t, vx2, label="vx2")
ax[1,0].set_title("vx1(t), vx2(t)")
ax[1,0].legend()

ax[1,1].plot(t, vy1, label="vy1")
ax[1,1].plot(t, vy2, label="vy2")
ax[1,1].set_title("vy1(t), vy2(t)")
ax[1,1].legend()

ax[2,0].plot(t, E_vals)
ax[2,0].set_title("Energía total vs t")

ax[2,1].plot(t, L_vals)
ax[2,1].set_title("Momento angular total vs t")

plt.tight_layout()
plt.savefig("1_c_resultados.pdf")