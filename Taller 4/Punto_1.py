import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pde import PDE, ScalarField, FieldCollection, CartesianGrid, MemoryStorage
# Clases principales del paquete 'pde': definición de EDPs, campos, mallas y almacenamiento en memoria


#import shutil
'''
ffmpeg_path = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg" # Busca ffmpeg o usa una ruta por defecto
mpl.rcParams["animation.writer"] = "ffmpeg" # Dice a Matplotlib que use ffmpeg como escritor
mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path # Fija la ruta al binario de ffmpeg encontrado
print("Usando ffmpeg en:", ffmpeg_path) # Confirma en consola qué ffmpeg se usará
'''

#Parametros
alpha = 0.1
x_min, x_max = -20.0, 20.0
N = 1001  # Número de nodos espaciales
dt_init = 1e-6  # dt inicial
dt_out = 0.05  # cada cuánto guardar frames para animación/observables

dx = (x_max-x_min)/(N-1)

# --------- Dominio 1D ----------
grid = CartesianGrid([(x_min, x_max)], shape=(N,)) # Crea una malla cartesiana 1D con N puntos en [x_min, x_max]
x = grid.axes_coords[0] # Vector de coordenadas x correspondiente a la malla

# --------- Función para hacer potencial ----------
def make_potential(grid, V): # Recibe la malla y una función Python V(x) que define el potencial
    x_vals = grid.axes_coords[0] # Extrae las coordenadas x de la malla
    data = np.asarray(V(x_vals)) # Evalúa V en todos los puntos de la malla y lo pasa a arreglo NumPy
    return ScalarField(grid, data=data) # Devuelve un campo escalar (ScalarField) con esos valores sobre la malla

# --------- Función de simulación ----------
def simulate_schrodinger(potential, potential_name, t_final, name):
    # Ejecuta la evolución temporal de la ecuación de Schrödinger (en forma real) y genera animación y observables.
    
    # Crear el potencial
    V_field = make_potential(grid, potential)
    
    # Definición de la PDE en forma real (u, v)
    bc_neumann = {"type": "derivative", "value": 0} # Condiciones de frontera de Neumann: derivada normal = 0

    # Define el sistema EDP acoplado para (u,v) con ψ = u + i v
    eq = PDE(
        {
            "u": "-alpha * laplace(v) + V * v", # Ecuación para u_t (parte real)
            "v": " alpha * laplace(u) - V * u", # Ecuación para v_t (parte imaginaria)
        },
        consts = {"alpha": alpha, "V": V_field}, # Pasa constantes y el campo del potencial a la EDP
        bc = {"u": bc_neumann, "v": bc_neumann}, # Aplica Neumann en ambos campos
    )

    # --------- Condición inicial ----------
    def psi0(x, x0=10.0, k0=2.0, width=2.0): # Define ψ(x,0) como gaussiana centrada en x0 con “ancho” y fase e^{-ik0 x}
        g = np.exp(-width * (x - x0)**2)
        return g * np.exp(-1j * k0 * x) # Multiplica por fase compleja => paquete con impulso hacia +x

    def norm(phi):
        norm = np.sum(np.square(np.abs(phi))) # Norma al cuadrado (suma de |ψ|^2 en la malla)
        return phi/np.sqrt(norm) # Devuelve ψ normalizado

    psi_ini = norm(psi0(x)) # Construye y normaliza la condición inicial en los nodos de la malla

    u0 = ScalarField(grid, data=psi_ini.real) # Parte real inicial u(x,0)
    v0 = ScalarField(grid, data=psi_ini.imag) # Parte imaginaria inicial v(x,0)
    state = FieldCollection([u0, v0], labels=["u", "v"]) # Empaqueta ambos campos en una colección 

    # --------- Almacenamiento durante la simulación ----------
    storage = MemoryStorage() # Estructura para guardar estados en tiempos específicos

    result = eq.solve( # Lanza el solver temporal de la EDP
        state, # Estado inicial (u0, v0)
        t_range = (0.0, t_final), # Intervalo temporal de simulación
        dt = dt_init, # Paso de tiempo inicial (el solver puede adaptarlo)
        tracker = ["progress", storage.tracker(interrupts=dt_out)], 
        # “progress” muestra barra/avance; el tracker de storage guarda un snapshot cada dt_out
    )

    # --------- Postproceso: observables y animaciones ----------
    # Extraer tiempos y estados
    times = np.asarray(storage.times)
    # Por si no guardó el estado final exacto, añadimos el actual:
    if times.size == 0 or times[-1] < t_final - 1e-12:
        storage.append(result, t_final) # Si el último estado no coincide con t_final, añade el actual a t_final
        times = np.array(storage.times) # Actualiza el vector de tiempos

    # Calcular μ(t) (media), σ(t) (desviación) a partir de ρ(x,t) = |ψ|^2
    mu_list, sig_list = [], [] #media y desviación 
    for _, st in storage.items(): # Recorre todos los estados guardados
        u = st["u"]; v = st["v"] # Recupera campos u y v en ese tiempo
        # Densidad de probabilidad ρ = u^2 + v^2
        rho = u**2 + v**2   
        mass = np.trapezoid(rho.data, x) #Norma^2
        # µ y σ
        rho_data = rho.data # Arreglo NumPy con los valores de ρ
        mu = np.trapezoid(x * rho_data, x) / mass # Media μ
        var = np.trapezoid((x - mu)**2 * rho_data, x) / mass # Varianza σ^2
        mu_list.append(mu) # Guarda μ(t)
        sig_list.append(np.sqrt(var)) # Guarda σ(t)

    mu_arr = np.array(mu_list) # Pasa listas a arreglos 
    sig_arr = np.array(sig_list)

    # --------- Animación de |ψ|^2 ----------
    fig, ax = plt.subplots(figsize=(10, 6)) # Crea figura y ejes para la animación de la densidad
    line, = ax.plot([], [], lw=2, label=r'$|\psi|^2$') # Línea vacía que se actualizará en cada frame
    ax.set_xlim(x_min, x_max) 
    # Escala del eje y basada en el máximo de densidad durante toda la simulación
    max_rho = 0
    for _, st in storage.items(): # Escanea todos los snapshots guardados
        u = st["u"]; v = st["v"]
        rho = u**2 + v**2
        max_rho = max(max_rho, np.max(rho.data)) # Actualiza el máximo observado de ρ
    ax.set_ylim(0.0, 1.2 * max_rho) # Margen superior
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi|^2$")
    
    # Graficar el potencial
    ax2 = ax.twinx() # Crea un segundo eje y para superponer el potencial
    V_vals = V_field.data # Extrae el vector de valores del potencial V(x)
    ax2.plot(x, V_vals, 'r--', alpha=0.7, label='$V(x)$')
    ax2.set_ylabel('Potencial $V(x)$')
    V_min, V_max = np.min(V_vals), np.max(V_vals) # Rango del potencial
    ax2.set_ylim(V_min, V_max)
    
    # Unir leyendas
    lines1, labels1 = ax.get_legend_handles_labels() # Toma leyendas del primer eje
    lines2, labels2 = ax2.get_legend_handles_labels() # Toma leyendas del segundo eje
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right') # Funde ambas en una sola leyenda
    
    writer = FFMpegWriter(fps=20, bitrate=2400) # Configura el escritor de video (20 fps, bitrate moderado)

    def frame_set(k): # Función que actualiza la figura para el frame k
        t_k = float(times[k]) # Tiempo correspondiente al frame k
        st = storage[k] # Estado guardado número k (FieldCollection)
        u_data = st[0].data # Parte real u en arreglo NumPy
        v_data = st[1].data # Parte imaginaria v en arreglo NumPy
        rho = u_data**2 + v_data**2 # Densidad |ψ|^2 para ese tiempo
        line.set_data(x, rho) # Actualiza datos de la línea en el gráfico
        ax.set_title(f"{potential_name}\nt = {t_k:.2f}   μ={mu_arr[k]:.2f}   σ={sig_arr[k]:.2f}")
        return (line,)

    K = min(len(times), len(mu_arr), len(sig_arr), len(storage)) #frames a generar
    with writer.saving(fig, f"{name}.mp4", dpi=120): # Abre contexto para escribir el video MP4
        for k in range(K): # Itera sobre todos los frames
            frame_set(k) # Actualiza la figura para el frame k
            writer.grab_frame() # Captura el frame actual y lo añade al video
    plt.close(fig)

    # --------- Curvas media y varianza ----------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    ax2.plot(times, mu_arr, lw=2, label=r"$\mu(t)$") # Traza la media 
    ax2.fill_between(times, mu_arr - sig_arr, mu_arr + sig_arr, alpha=0.3, label=r"$\mu\pm\sigma$") # Banda de confianza (desviación)
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$\mu$")
    ax2.set_title(fr"$\mu\pm\sigma$ - {potential_name}")
    ax2.legend()
    
    fig2.tight_layout()
    fig2.savefig(f"{name}.pdf", dpi=160)
    plt.close(fig2)

# --------- Ejecutar las tres simulaciones ----------

# 1.a Oscilador armónico
def V_harmonic(x):
    return (x**2)/50

simulate_schrodinger(V_harmonic, "Oscilador armónico", 150.0, "1.a")

# 1.b Oscilador cuártico
def V_quartic(x):
    return (x/5)**4

simulate_schrodinger(V_quartic, "Oscilador cuártico", 50.0, "1.b")

# 1.c Potencial sombrero
def V_hat(x):
    return ((x**4)/100 - (x**2))/50

simulate_schrodinger(V_hat, "Potencial sombrero", 150.0, "1.c")