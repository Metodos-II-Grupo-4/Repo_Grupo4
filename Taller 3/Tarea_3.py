import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.animation as animation
from joblib import Parallel, delayed

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

plt.savefig("1.c.pdf")

#2. Balistica

m = 10.01 #kg
g = 9.773 #m/s2

#Coeficiente de friccion 
def beta(y, A=1.642, B=40.624, C=2.36):
    k = max(1.0 - y / B, 0.0)
    return A*(k**C)

#2.a. Alcance
def dynamics(t, Y, theta, v0):
    x, y, vx, vy = Y #vector de estado 
    
    if t == 0:
        vx = v0 * np.cos(theta) #componentes de la velocidad
        vy = v0 * np.sin(theta)
        
    v = np.hypot(vx, vy) #magnitud
    
    dxdt = vx
    dydt = vy
    dvxdt = -(beta(y)/m)*v*vx #Fuerza por la friccion 
    dvydt = -g - (beta(y)/m)*v*vy #Fuerza por la friccion + gravedad
    
    return np.array([dxdt, dydt, dvxdt, dvydt]) #derivada temporal

# Define una función que detecta cuándo el proyectil toca el suelo
def piso(t, Y, theta, v0):
    x, y, vx, vy = Y
    return y # Retorna la altura y (cuando y = 0, el proyectil toca el suelo)
piso.terminal = True #la integración debe terminar cuando se cumpla la condición del evento piso
piso.direction = -1 # Especifica que el evento se activa cuando la altura y cruza de positivo a negativo

# Define una función que calcula la trayectoria completa del proyectil
def solucion(theta, v0):
    theta = np.deg2rad(theta) # grados a radianes
    Y_inicial = np.array([0, 0, v0*np.cos(theta), v0*np.sin(theta)]) # vector de estado inicial
    
    solv = solve_ivp(
        fun=dynamics, 
        t_span=(0,50.),       # Intervalo de tiempo de integración [0, 50] segundos
        y0=Y_inicial, 
        max_step=0.01,        # Tamaño máximo de paso para la integración
        args=(v0, theta),     # Argumentos adicionales para la función dynamics
        events=piso,          # Función de evento para detectar impacto en el suelo
        dense_output=True,    # Permite interpolación de la solución
        method='RK45',        # Método de integración: Runge-Kutta de orden 4(5)
        rtol=1e-6, atol=1e-9) # Tolerancias relativas y absolutas para la precisión
    return solv

x_solucion = lambda theta, v0: solucion(theta, v0).y_events[0][0][0] #posición x del punto de impacto con el suelo

# ---------------------- Derivadas numéricas (centradas) ----------------------
def derivada(theta, v0, h=1e-3): #1ra derivada
    return (x_solucion(theta + h, v0) - x_solucion(theta - h, v0)) / (2*h)

def dderivada(theta, v0, h=1e-3): #2nda derivada
    return (x_solucion(theta + h, v0) - 2*x_solucion(theta, v0) + x_solucion(theta - h, v0)) / (h**2)

# Newton-Raphson para encontrar el ángulo que maximiza el alcance
def Newton_Raphson(v0, theta0=45.0, bdd=(10.0,80.0), h=1e-3, max_iter=50, tol=1e-4):
    theta = float(np.clip(theta0, bdd[0], bdd[1])) #Tomar un theta inicial entre el rango (10, 80)
    x_theta = x_solucion(theta, v0) # Alcance inicial
    for k in range(max_iter):
        f1 = derivada(theta, v0, h) #1ra derivada
        f2 = dderivada(theta, v0, h) #2nda derivada

        step = - f1/f2   # Paso de Newton: -f'(θ)/f''(θ) para maximizar el alcance
        theta_new = theta + step # Propone un nuevo ángulo sumando el paso
        max_step = 5.0  # grados
        if step > max_step: # Limita el paso si es demasiado grande positivo
            step = max_step
        if step < -max_step: # Limita el paso si es demasiado grande negativo
            step = -max_step
        
        # Calcula el nuevo ángulo con el paso limitado y asegura que esté dentro de los límites
        theta_new = float(np.clip(theta + step, bdd[0], bdd[1])) 
        x_new = x_solucion(theta_new, v0)
        # si no mejora, hacer un pequeño paso
        if x_new < x_theta - 1e-12:
            small_step = np.sign(f1) * 0.5 # Toma un paso pequeño en la dirección de la derivada
            theta_new = float(np.clip(theta + small_step, bdd[0], bdd[1]))
            x_new = x_solucion(theta_new, v0)
            if x_new < x_theta: # Si aún no mejora, termina la optimización
                break
        theta, x_theta = theta_new, x_new # Actualiza theta y x_theta
    return theta, x_theta, k+1 # Retorna el ángulo óptimo, el alcance máximo y el número de iteraciones

v0_values = np.linspace(5.0, 80.0, 25)
x_max = []
angles = []

theta = 45.0 
for v0 in v0_values:
    theta_nr, x_nr,_ = Newton_Raphson(v0, theta) #theta se actualiza con el angulo antes encontrado 
    x_max.append(x_nr)
    angles.append(theta_nr)
    theta = angles[-1]

print('hola')
plt.figure(figsize=(10, 6))
plt.plot(v0_values, x_max, marker='o', linewidth=1)
plt.xlabel(r'$v_0$ (m/s)')
plt.ylabel(r'$x_{max}$ (m)')
plt.title('Alcance máximo horizontal vs velocidad inicial')
plt.grid(True)
plt.savefig("2.a.pdf")
print('hola :0')

#BONO
#--------------------------

#2.b Atinar a un objetivo

# Define una función que encuentra el ángulo para golpear un objetivo específico
def angle_to_hit_target(v0, target_x, target_y):
    def objective(theta):
        sol = solucion(theta, v0) # Calcula la trayectoria del proyectil para un ángulo theta dado
        x_traj = sol.y[0] #coordenadas x de la trayectoria
        y_traj = sol.y[1] #coordenadas x de la trayectoria
        distances = np.sqrt((x_traj - target_x)**2 + (y_traj - target_y)**2) #distancia desde cada punto hasta el objetivo
        return np.min(distances)
    
    res = minimize_scalar(objective, bounds=(10., 80.), method='bounded', options={'xatol': 1e-3, 'maxiter': 80})
    #minimize_scalar de scipy
    # Límites del ángulo de búsqueda (10° a 80°)
    # Método de optimización para problemas con límites
    # Tolerancia en el ángulo (1e-3 grados)
    # Número máximo de iteraciones
    if res.fun > 0.1: # Verifica si la distancia mínima encontrada es aceptable (menor a 0.1 metros)
        return None
    return np.round(res.x, 2) # Retorna el ángulo óptimo redondeado a 2 decimales

#2.c Varias opciones para disparar

v0_values = np.linspace(10, 140, 100)  # Velocidades iniciales
theta0_values = np.linspace(10, 80, 100)  # Angulos iniciales
soluciones = []

for v0 in v0_values:
    for theta0 in theta0_values:
        sol = solucion(theta0, v0) #determina las trayectorias
        x_traj = sol.y[0] #compomente x 
        y_traj = sol.y[1] # componente y
        distances = np.sqrt((x_traj - 12.)**2 + (y_traj - 0.)**2) #determina distancia entre la trayectoria y el punto (12, 0)
        if np.min(distances) < 0.1: #si se acerca por 0.1m se considera que le atino
            soluciones.append((v0, theta0)) 
soluciones = np.array(soluciones)

plt.figure(figsize=(10, 6))
plt.scatter(soluciones[:, 0], soluciones[:, 1], s=10)
plt.xlabel(r'$v_0$ (m/s)')
plt.ylabel(r'$\theta_0$ (degrees)')
plt.title('Condiciones iniciales (12m, 0)')
plt.grid(True)
plt.savefig('2.c.pdf')

# PUNTO 3 (Molécula diatómica)

hbar, a, x0 = 0.1, 0.8, 10.0
dx = 0.005

V = lambda x: (1.0 - np.exp(a*(x - x0)))**2 - 1.0
# Se define la Energía de Morse, como su mínimo es -1, las E ligadas estan (-1,0)

def x2_turn(eps):
    s = np.sqrt(1.0 + eps)
    return x0 + np.log(1.0 + s)/a

# Se fija el borde derecho desde el cual se va a simular

def integrate(eps, slope0=1e-6):
    xL = 0.0
    xR = x2_turn(eps) + 1.0
    N = int(np.ceil((xR - xL)/dx)) + 1
    x = np.linspace(xL, xR, N)
    psi = np.zeros(N)
    dpsi = np.zeros(N)
    psi[0] = 0.0
    dpsi[0] = slope0
    inv_h2 = 1.0/(hbar*hbar)
    for i in range(N-1):
        xi = x[i]
        Vi = V(xi)
        k1p, k1q = dpsi[i], (Vi - eps)*psi[i]*inv_h2
        xh = xi + 0.5*dx
        Vh = V(xh)
        ph = psi[i] + 0.5*dx*k1p
        qh = dpsi[i] + 0.5*dx*k1q
        k2p, k2q = qh, (Vh - eps)*ph*inv_h2
        ph2 = psi[i] + 0.5*dx*k2p
        qh2 = dpsi[i] + 0.5*dx*k2q
        k3p, k3q = qh2, (Vh - eps)*ph2*inv_h2
        x1s = xi + dx
        V1 = V(x1s)
        p1 = psi[i] + dx*k3p
        q1 = dpsi[i] + dx*k3q
        k4p, k4q = q1, (V1 - eps)*p1*inv_h2
        psi[i+1]  = psi[i]  + (dx/6.0)*(k1p + 2*k2p + 2*k3p + k4p)
        dpsi[i+1] = dpsi[i] + (dx/6.0)*(k1q + 2*k2q + 2*k3q + k4q)
        if not np.isfinite(psi[i+1]) or not np.isfinite(dpsi[i+1]):
            return None, None, 1e50
    return x, psi, np.hypot(psi[-1], dpsi[-1])
# Integrador de Shrodinger. Psi'' = (V - E)*psi/hbar^2. p = psi, q = psi'
# q' = (V - E)*p/hbar^2. psi(0)=0, psi'(0)=slope0 pequeño
# k1 ... k4 son los incrementos de Runge-Kutta. Al final del bucle se alcanza x_i+1
# en caso de nan o inf, se retorna 1e50 para que no se tome como mínimo
# retorna malla x, funcion psi y la norma al borde derecho

def golden_min(aE, bE, iters=22):
    phi = (1 + np.sqrt(5.0))/2.0
    inv = 1.0/phi
    L, R = aE, bE
    c = R - (R - L)*inv
    d = L + (R - L)*inv
    fc = integrate(c)[2]
    fd = integrate(d)[2]
    for _ in range(iters):
        if fc < fd: 
            R, d, fd = d, c, fc
            c = R - (R - L)*inv
            fc = integrate(c)[2]
        else:       
            L, c, fc = c, d, fd
            d = L + (R - L)*inv
            fd = integrate(d)[2]
    return 0.5*(L+R)
# Minimiza la norma de borde sin utilizar derivadas

E_scan = np.linspace(-0.99, -0.01, 1400)
norms = np.array([integrate(E)[2] for E in E_scan])
sm = np.convolve(np.pad(np.log10(norms), (4,4), 'edge'), np.ones(9)/9.0, 'valid')
mins = [i for i in range(1, len(sm)-1) if sm[i] < sm[i-1] and sm[i] < sm[i+1]]
# Barrido de energías y preselección de mínimos
# norms: valor de la norma ||(psi, psi')|| en el borde para cada E
# np.log10: se usa para reducir el rango dinámico
# sm: suavizado de norms con una ventana de 9 puntos
# convolve: promedio móvil 
# mins: índices donde el suavizado tiene un mínimo local estricto 


E_list, states, last = [], [], -999
for i in mins:
    if i - last < 16: continue
    last = i
    L, R = max(0, i-8), min(len(E_scan)-1, i+8)
    e = golden_min(E_scan[L], E_scan[R])
    x, psi, _ = integrate(e)
    if x is None: continue
    A = np.trapz(psi*psi, x)
    if A <= 0 or not np.isfinite(A): continue
    states.append((x, psi/np.sqrt(A)))
    E_list.append(e)
# Filtro de separación i - last < 16: para evitar tomar limites demasiado cercanos
# Se aplica goldenmin en un intervalo reducido cercano al mínimo en cuestión
# Se integra la energía y se normaliza usanto trapezoide
# se guarda la pareja (x, psi_normalizada) y la energía

ordr = np.argsort(E_list)
E_list = list(np.array(E_list)[ordr])
states = [states[i] for i in ordr]
plt.figure(figsize=(7.2,5.2), dpi=130)
for E,(x,psi) in zip(E_list, states):
    plt.plot(x, E + 0.12*psi, lw=1.2)
    plt.hlines(E, 0, x[-1], ls='dotted', lw=0.8)
xg = np.linspace(0, 12, 1200)
plt.plot(xg, V(xg), 'k', lw=1.0, alpha=0.9, label="Morse potential")
plt.xlim(0,12)
plt.ylim(-1.02,0.05)
plt.xlabel("x")
plt.ylabel("Energy")
plt.legend(loc="lower left", frameon=False)
plt.tight_layout()
plt.savefig("3.pdf", bbox_inches="tight")
with open("3.txt","w",encoding="utf-8") as f:
    for k,E in enumerate(E_list): f.write(f"n={k}\tE={E:.8f}\n")
# Graficar cada psi_n normalizada separada verticalmente (con un factor de escala 0.12 por razones visuales)
# con hlines se grafica la energía correspondiente.
# En negro se traza la referencia (potencial de Morse)
# Fija límites y guarda en PNG, escribe un archivo de texto con las energías halladas

# PUNTO 4
def F_4(t, Y, alpha=1.0):
    theta, r, Ptheta, Pr = Y #Son dos ecuaciones de segundo orden, se necesitan 4 entradas en el vector de estado.
    return np.array([Ptheta/(r+1)**2,
                     Pr,
                     -alpha**2 * (r+1) * np.sin(theta),
                     alpha**2 * np.cos(theta) - r + Ptheta**2 / (1+r)**3])
    
def cruzar_cero(t, Y, *args):
    theta, r, Ptheta, Pr = Y
    return theta

cruzar_cero.terminal = False
cruzar_cero.direction = 0
    
alphas = np.linspace(1, 1.2, 20)
fig, ax = plt.subplots(1,1)
t = np.linspace(0, 10000, 10000)
ax.set_xlabel("r")
ax.set_ylabel("P_r")

def simular_alpha(alpha):
    solucion_4 = solve_ivp(
        fun=F_4, #Pasamos la función de vector de estado
        args=(alpha,),
        t_span=(0, 10000), #Tiempo de simulación 10^4 s
        y0=np.array([np.pi/2, 0., 0., 0.]), #Valores iniciales theta=pi/2, r=0, dtheta/dt = 0 y dr/dt = 0.
        max_step=1, #dt=1
        events=cruzar_cero,
        dense_output=False,
        atol=1e-6, rtol=1e-6,
        method="RK45" #Usamos el método explícito Runge Kutta de orden 5(4)
    ) 
    return solucion_4.y_events[0]

resultados = Parallel(n_jobs=-1)(delayed(simular_alpha)(a) for a in alphas)
fig, ax = plt.subplots()
for cruces in resultados:
    for cruce in cruces:
        theta, r, Ptheta, Pr = cruce
        ax.scatter(r, Pr, s=1)

ax.set_xlabel("r")
ax.set_ylabel("P_r")
plt.savefig("4.pdf")

# PUNTO 5 (Circuito genético)

from scipy.signal import find_peaks

def func(t, y, a, b): #Se esta planteando el sistema de ecuaciones ciclico, donde p1 depende de p2, p2 de p3 y p3 de p1
    m1, m2, m3, p1, p2, p3 = y
    al0 = a/ 1000
    dm1 = a/(1 + p3**2) + al0 - m1
    dm2 = a/(1 + p1**2) + al0 - m2
    dm3 = a/(1+ p2**2) + al0 - m3
    dp1 = -b *(p1 - m1)
    dp2 = -b *(p2 - m2)
    dp3 = -b*(p3 - m3)

    return [dm1, dm2, dm3, dp1,dp2, dp3 ] # Devuelve las soluciones, m son la cantidad de moleculas de ARNm y p son la cantidad de proteinas traducidas por ARNm


def calculate_amplitude(a, b):
    y0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    t_eval = np.linspace(0, 400, 700)

    sol = solve_ivp(func, (0,400), y0, args=(a, b), t_eval=t_eval, method='RK45')
    p3_values = sol.y[5] 
    # Agarra p3, que esta en la 6 columna de las soluciones
    p3_max, _  = find_peaks(p3_values)
    p3_min, _ = find_peaks(-p3_values)
    maxi = np.max(p3_values[p3_max])
    mini = np.min(p3_values[p3_min])
    amplitude = (maxi - mini)
    return np.log10(amplitude)
#Aqui queremos la amplitud de los picos cuando se vuelve estable, hallar el pico máximo y el mínimo, restarlos y esa sería la amplitud, se hace para cada conjunto de a y b


alpha_vals = np.logspace(0, 5, 5)
beta_vals = np.logspace(0, 3, 3)
amplitudes = np.zeros((len(alpha_vals), len(beta_vals))) # Matriz

for i, alpha in enumerate(alpha_vals): #Con enumerate queda mas comodo, para evaluar todas las posibles combinaciones de a y b en el logspace.
    for j, beta in enumerate(beta_vals):
        amplitudes[i, j] = calculate_amplitude(alpha, beta) # Hace la amplitud para cada conjunto de a y b

plt.figure(figsize=(10, 8))
X, Y = np.meshgrid(np.log10(alpha_vals), np.log10(beta_vals))
plt.contourf(X, Y, amplitudes.T, levels=20, cmap='hot')
plt.colorbar(label='log10(Amplitud de p3)(creciente)')
plt.xlabel('log10(α)')
plt.ylabel('log10(β)')
plt.title('Amplitud de Oscilación de p3 circuito genético.')
plt.tight_layout()
plt.savefig('5.pdf')


