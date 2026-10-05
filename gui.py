"""
Interfaz gráfica (tkinter) de EpiSim para explorar y entender el simulador.

Pestañas:
    - Paso a paso: una simulación día por día. La población se dibuja como una cuadrícula
      (una celda por persona, coloreada por estado), junto con la curva SEIR y la bitácora
      de cada número pseudoaleatorio usado en el día.
    - Montecarlo: N simulaciones del escenario (o sin vs. con vacunación), gráficas y tabla
      de estadísticas descriptivas.
    - Sensibilidad: análisis un parámetro a la vez (tornado).
    - Números pseudoaleatorios: validación de los R_i usados y de la independencia entre flujos.
    - ¿Cómo funciona?: resumen del modelo.

Los parámetros del panel izquierdo se usan en todas las pestañas. Se pueden cargar y guardar
en archivos .csv / .txt (mismo formato que config/).

Uso:
    venv\\Scripts\\python.exe gui.py

La lógica del modelo está en modelo.py, montecarlo.py y analisis.py; este archivo solo
contiene la interfaz.
"""
import os
import queue
import threading
import traceback

import matplotlib

matplotlib.use("Agg")   # Las figuras se incrustan en la ventana con FigureCanvasTkAgg

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

import analisis
import graficas
from aleatorio import (GENERADORES, FlujoAleatorio, crear_generador, saltar, semilla_flujo,
                       validar_independencia_flujos, validar_secuencia)
from configuracion import Configuracion, PARAMETROS_OPCION, cargar_configuracion
from modelo import E, I, NOMBRES_ESTADOS, R, S, Bitacora, SimulacionPasoAPaso
from montecarlo import ejecutar_montecarlo

DIR_BASE = os.path.dirname(os.path.abspath(__file__))

ETIQUETAS_RANGOS = [
    ("beta", "β  contactos/día"),
    ("p_transmision", "p  transmisión"),
    ("periodo_incubacion", "Incubación (días)"),
    ("periodo_infeccioso", "Infeccioso (días)"),
    ("cobertura_vacunacion", "Cobertura vacunación"),
    ("efectividad_vacuna", "Efectividad vacuna"),
    ("letalidad", "Letalidad"),
]
ETIQUETAS_ENTEROS = [
    ("poblacion", "Población (N)"),
    ("infectados_iniciales", "Infectados iniciales"),
    ("dias", "Días"),
    ("semilla_maestra", "Semilla maestra"),
    ("umbral_control", "Umbral de control"),
]
ETIQUETAS_OPCIONES = [
    ("modo_contactos", "Contactos"),
    ("seleccion_contacto", "Selección contacto"),
    ("generador", "Generador"),
]

# Cuadrícula de la población: código → (color, etiqueta)
CATEGORIAS = [
    ("#cde2fb", "S no vacunado"),
    ("#2a78d6", "S vacunado"),
    ("#eb6834", "E expuesto"),
    ("#1baf7a", "I infeccioso"),
    ("#e0b24a", "R recuperado"),
    ("#2b2b2b", "Fallecido"),
]
MAPA_COLORES = ListedColormap([c for c, _ in CATEGORIAS])
MAPA_COLORES.set_bad("white")

GRAFICAS_ESCENARIO = [
    "Curva epidémica",
    "Magnitud del pico",
    "Día del pico",
    "Total de infectados",
    "Fallecidos",
    "Días hasta control",
    "Parámetros muestreados",
    "Validación de los R_i",
    "Sensibilidad global (Spearman)",
]
GRAFICAS_COMPARACION = ["Comparación: curvas", "Comparación: distribuciones"]

TEXTO_AYUDA = """\
MODELO SEIR ESTOCÁSTICO BASADO EN INDIVIDUOS

Cada una de las N personas está en uno de cuatro estados:
  S  Susceptible        puede contagiarse
  E  Expuesto            infectado, todavía no contagia (período de incubación)
  I  Infeccioso          contagia a otros (período infeccioso)
  R  Recuperado/removido inmune o fallecido

INICIALIZACIÓN (una vez por simulación)
  1. Se muestrean los parámetros de la simulación, un R_i cada uno:
       β, p, cobertura, efectividad, letalidad  =  a + (b − a)·R      (transformación uniforme)
  2. Se eligen los casos índice: individuo = piso(R·N); a cada uno se le asigna su período infeccioso.
  3. Vacunación: para cada persona se genera un R; si R < cobertura queda vacunada y su
     susceptibilidad se multiplica por (1 − efectividad).

CADA DÍA
  4. Progresión (calendario de eventos, sin azar salvo la letalidad):
       I → R  a quienes se les cumplió el período infeccioso; R < letalidad → fallece
       E → I  a quienes se les cumplió la incubación; R → período infeccioso = a + piso(R·(b − a + 1))
  5. Paso 1 - contactos de cada infeccioso:  contactos = piso(R + β)
       (vale 0 o 1 con β < 1; su promedio es exactamente β contactos por día)
  6. Paso 2 - selección:  contactado = piso(R·N). Si no es susceptible, el contacto no contagia.
  7. Paso 3 - transmisión:  si R < p · factor_vacunación → el contactado pasa a E
       y R → su período de incubación.

NÚMEROS PSEUDOALEATORIOS
  Todos los R salen de un generador congruencial de pseudo_gen (Congruencial Lineal
  a = 1664525, c = 1013904223, m = 2^32, por defecto). Cada simulación usa su propio
  "flujo": la semilla maestra avanzada k·SALTO posiciones con la fórmula de salto
  adelante, así ninguna simulación comparte números con otra.
  En Montecarlo, los R usados por cada simulación se someten a las 5 pruebas
  (medias, varianza, χ², KS, póker); si falla alguna, la simulación se descarta
  y se usa el flujo siguiente.

MÉTRICAS
  Tasa de ataque   proporción de la población que se infectó
  Pico             máximo de I(t) y el día en que ocurre
  Día de control   primer día después del pico con E + I ≤ umbral de control
  R0 aproximado    β · p · (duración media del período infeccioso)

¿QUÉ PROBAR?
  • Paso a paso: comparar cobertura 0 con cobertura 0.7 usando el mismo flujo.
  • Contactos "literal" (piso(R·β)): la epidemia no avanza, porque piso(R·β) = 0 si β < 1.
  • Selección "susceptibles": la epidemia no se frena al agotarse los susceptibles.
  • β entre 0.9 y 1.5: aparecen días con 2 contactos por infeccioso.
"""


# ── Utilidades de la interfaz ───────────────────────────────────────────

class PanelFigura(ttk.Frame):
    """Marco que muestra una figura de Matplotlib con su barra de herramientas."""

    def __init__(self, padre, **kw):
        super().__init__(padre, **kw)
        self.figura = None
        self.canvas = None
        self.barra = None
        self.vacio = ttk.Label(self, text="Aún no hay gráfica.", foreground="gray", anchor="center")
        self.vacio.pack(fill="both", expand=True)

    def mostrar(self, fig):
        self.limpiar()
        if fig is None:
            self.vacio.pack(fill="both", expand=True)
            return
        self.vacio.pack_forget()
        fig.set_layout_engine("tight")   # Reajusta márgenes cuando cambia el tamaño de la ventana
        self.figura = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.barra = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.barra.update()
        self.barra.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.draw()

    def limpiar(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.barra.destroy()
            plt.close(self.figura)
            self.canvas = self.barra = self.figura = None


def crear_tabla(padre, columnas, anchos, alto=8):
    """Treeview con barra de desplazamiento."""
    marco = ttk.Frame(padre)
    tabla = ttk.Treeview(marco, columns=columnas, show="headings", height=alto)
    for col, ancho in zip(columnas, anchos):
        tabla.heading(col, text=col)
        tabla.column(col, width=ancho, anchor="center" if ancho < 150 else "w", stretch=True)
    barra = ttk.Scrollbar(marco, orient="vertical", command=tabla.yview)
    tabla.configure(yscrollcommand=barra.set)
    tabla.pack(side="left", fill="both", expand=True)
    barra.pack(side="right", fill="y")
    return marco, tabla


def formato_numero(valor):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    if abs(valor) < 1:
        return f"{valor:.4f}"
    return f"{valor:,.1f}"


# ── Aplicación ──────────────────────────────────────────────────────────

class Aplicacion(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("EpiSim - Simulación Montecarlo SEIR")
        self.geometry("1500x900")
        self.minsize(1150, 720)

        self.cola = queue.Queue()     # Mensajes de los hilos de trabajo → interfaz
        self.trabajando = False
        self.archivo_semillas = ""

        # Estado de la pestaña "Paso a paso"
        self.sim = None
        self.flujo_actual = None
        self.reproduciendo = False

        # Resultados de Montecarlo: nombre → ResultadosEscenario
        self.resultados = {}
        self.sensibilidad = None

        estilo = ttk.Style(self)
        if "vista" in estilo.theme_names():
            estilo.theme_use("vista")
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 10, "bold"))
        estilo.configure("Dato.TLabel", font=("Segoe UI", 10))

        principal = ttk.Panedwindow(self, orient="horizontal")
        principal.pack(fill="both", expand=True)
        izquierda = ttk.Frame(principal, padding=8)
        derecha = ttk.Frame(principal, padding=(0, 8, 8, 8))
        principal.add(izquierda, weight=0)
        principal.add(derecha, weight=1)

        self._construir_panel_parametros(izquierda)
        self.pestanas = ttk.Notebook(derecha)
        self.pestanas.pack(fill="both", expand=True)
        self._construir_paso_a_paso()
        self._construir_montecarlo()
        self._construir_sensibilidad()
        self._construir_numeros()
        self._construir_ayuda()

        self.barra_estado = ttk.Label(self, text="Listo.", relief="sunken", anchor="w", padding=(6, 2))
        self.barra_estado.pack(side="bottom", fill="x")

        ruta = os.path.join(DIR_BASE, "config", "con_vacunacion.csv")
        if os.path.exists(ruta):
            self._aplicar_configuracion(cargar_configuracion(ruta))
        else:
            self._aplicar_configuracion(Configuracion())
        self.after(100, self._revisar_cola)

    # ── Panel de parámetros ─────────────────────────────────────────────

    def _construir_panel_parametros(self, marco):
        ttk.Label(marco, text="Parámetros del escenario", style="Titulo.TLabel").pack(anchor="w")
        botones = ttk.Frame(marco)
        botones.pack(fill="x", pady=(4, 8))
        ttk.Button(botones, text="Cargar…", command=self._cargar_archivo).pack(side="left")
        ttk.Button(botones, text="Guardar…", command=self._guardar_archivo).pack(side="left", padx=4)
        ttk.Button(botones, text="Enunciado", command=lambda: self._aplicar_configuracion(Configuracion())
                   ).pack(side="left")

        self.var = {}
        grupo = ttk.LabelFrame(marco, text="General", padding=6)
        grupo.pack(fill="x")
        self.var["nombre"] = tk.StringVar()
        ttk.Label(grupo, text="Nombre").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(grupo, textvariable=self.var["nombre"], width=20).grid(row=0, column=1, sticky="we", pady=2)
        for fila, (clave, etiqueta) in enumerate(ETIQUETAS_ENTEROS, start=1):
            self.var[clave] = tk.StringVar()
            ttk.Label(grupo, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=2)
            ttk.Entry(grupo, textvariable=self.var[clave], width=20).grid(row=fila, column=1, sticky="we", pady=2)

        grupo = ttk.LabelFrame(marco, text="Distribuciones U(mín, máx)", padding=6)
        grupo.pack(fill="x", pady=8)
        ttk.Label(grupo, text="mín", foreground="gray").grid(row=0, column=1)
        ttk.Label(grupo, text="máx", foreground="gray").grid(row=0, column=2)
        for fila, (clave, etiqueta) in enumerate(ETIQUETAS_RANGOS, start=1):
            a, b = tk.StringVar(), tk.StringVar()
            self.var[clave] = (a, b)
            ttk.Label(grupo, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=2, padx=(0, 6))
            ttk.Entry(grupo, textvariable=a, width=8).grid(row=fila, column=1, pady=2, padx=1)
            ttk.Entry(grupo, textvariable=b, width=8).grid(row=fila, column=2, pady=2, padx=1)
        ttk.Label(grupo, text="Valor fijo: mismo número en mín y máx.\nPeríodos en días enteros.",
                  foreground="gray").grid(row=len(ETIQUETAS_RANGOS) + 1, column=0, columnspan=3,
                                          sticky="w", pady=(4, 0))

        grupo = ttk.LabelFrame(marco, text="Opciones del modelo", padding=6)
        grupo.pack(fill="x")
        for fila, (clave, etiqueta) in enumerate(ETIQUETAS_OPCIONES):
            self.var[clave] = tk.StringVar()
            ttk.Label(grupo, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=2, padx=(0, 6))
            ttk.Combobox(grupo, textvariable=self.var[clave], values=PARAMETROS_OPCION[clave],
                         state="readonly", width=14).grid(row=fila, column=1, pady=2)
        self.var["alpha"] = tk.StringVar()
        ttk.Label(grupo, text="α pruebas").grid(row=len(ETIQUETAS_OPCIONES), column=0, sticky="w", pady=2)
        ttk.Entry(grupo, textvariable=self.var["alpha"], width=16).grid(row=len(ETIQUETAS_OPCIONES), column=1,
                                                                         pady=2)

        self.lbl_origen = ttk.Label(marco, text="", foreground="gray", wraplength=260)
        self.lbl_origen.pack(anchor="w", pady=(8, 0))

    def _aplicar_configuracion(self, cfg):
        """Copia una Configuracion a los campos del panel."""
        for clave, var in self.var.items():
            valor = cfg[clave]
            if isinstance(var, tuple):
                var[0].set(f"{valor[0]:g}")
                var[1].set(f"{valor[1]:g}")
            else:
                var.set(str(valor))
        self.archivo_semillas = cfg["archivo_semillas"]
        self.lbl_origen.config(text=f"Origen: {cfg.origen}")

    def leer_configuracion(self, **cambios):
        """Construye una Configuracion con los valores del panel (lanza ValueError si hay errores)."""
        valores = {}
        for clave, var in self.var.items():
            if isinstance(var, tuple):
                valores[clave] = f"U({var[0].get()}, {var[1].get()})"
            else:
                valores[clave] = var.get()
        valores["archivo_semillas"] = self.archivo_semillas
        valores.update(cambios)
        return Configuracion(valores, origen="interfaz gráfica")

    def _cargar_archivo(self):
        ruta = filedialog.askopenfilename(initialdir=os.path.join(DIR_BASE, "config"),
                                          filetypes=[("Configuración", "*.csv *.txt"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            self._aplicar_configuracion(cargar_configuracion(ruta))
            self._estado(f"Configuración cargada: {ruta}")
        except (ValueError, FileNotFoundError) as e:
            messagebox.showerror("Error al cargar", str(e))

    def _guardar_archivo(self):
        try:
            cfg = self.leer_configuracion()
        except ValueError as e:
            messagebox.showerror("Parámetros inválidos", str(e))
            return
        ruta = filedialog.asksaveasfilename(initialdir=os.path.join(DIR_BASE, "config"),
                                            initialfile=f"{cfg['nombre']}.csv", defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if ruta:
            cfg.guardar(ruta)
            self._estado(f"Configuración guardada en {ruta}")

    # ── Pestaña: Paso a paso ────────────────────────────────────────────

    def _construir_paso_a_paso(self):
        pestana = ttk.Frame(self.pestanas, padding=6)
        self.pestanas.add(pestana, text="Paso a paso")

        controles = ttk.Frame(pestana)
        controles.pack(fill="x")
        ttk.Label(controles, text="Flujo (simulación nº)").pack(side="left")
        self.var_flujo = tk.StringVar(value="1")   # El flujo 1 muestra un brote grande
        ttk.Spinbox(controles, from_=0, to=2046, textvariable=self.var_flujo, width=6).pack(side="left", padx=4)
        ttk.Button(controles, text="Nueva simulación", command=self._nueva_simulacion).pack(side="left", padx=4)
        self.btn_dia = ttk.Button(controles, text="Siguiente día ▸", command=self._siguiente_dia)
        self.btn_dia.pack(side="left", padx=2)
        self.btn_reproducir = ttk.Button(controles, text="▶ Reproducir", command=self._alternar_reproduccion)
        self.btn_reproducir.pack(side="left", padx=2)
        ttk.Button(controles, text="Ir al final ⏭", command=self._ir_al_final).pack(side="left", padx=2)
        ttk.Label(controles, text="   Velocidad (días/s)").pack(side="left")
        self.var_velocidad = tk.DoubleVar(value=8)
        ttk.Scale(controles, from_=1, to=40, variable=self.var_velocidad, length=140).pack(side="left", padx=4)
        self.lbl_semilla = ttk.Label(controles, text="", foreground="gray")
        self.lbl_semilla.pack(side="left", padx=10)

        self.lbl_conteo = ttk.Label(pestana, text="Presione «Nueva simulación».", style="Dato.TLabel")
        self.lbl_conteo.pack(anchor="w", pady=(6, 0))
        self.lbl_params = ttk.Label(pestana, text="", foreground="#52514e")
        self.lbl_params.pack(anchor="w")

        divisor = ttk.Panedwindow(pestana, orient="vertical")
        divisor.pack(fill="both", expand=True, pady=(4, 0))

        marco_fig = ttk.Frame(divisor)
        self.fig_paso = Figure(figsize=(10, 4.2))
        self.ax_grid = self.fig_paso.add_subplot(1, 2, 1)
        self.ax_curva = self.fig_paso.add_subplot(1, 2, 2)
        self.fig_paso.set_layout_engine("tight")
        self.canvas_paso = FigureCanvasTkAgg(self.fig_paso, master=marco_fig)
        self.canvas_paso.get_tk_widget().pack(fill="both", expand=True)
        divisor.add(marco_fig, weight=3)

        marco_log = ttk.Frame(divisor)
        ttk.Label(marco_log, text="Bitácora: cada número pseudoaleatorio usado y para qué se usó",
                  style="Titulo.TLabel").pack(anchor="w")
        interno = ttk.Frame(marco_log)
        interno.pack(fill="both", expand=True)
        self.txt_log = tk.Text(interno, height=10, wrap="none", font=("Consolas", 9))
        barra_y = ttk.Scrollbar(interno, orient="vertical", command=self.txt_log.yview)
        barra_x = ttk.Scrollbar(marco_log, orient="horizontal", command=self.txt_log.xview)
        self.txt_log.configure(yscrollcommand=barra_y.set, xscrollcommand=barra_x.set)
        self.txt_log.pack(side="left", fill="both", expand=True)
        barra_y.pack(side="right", fill="y")
        barra_x.pack(fill="x")
        self.txt_log.tag_configure("dia", font=("Consolas", 9, "bold"))
        self.txt_log.tag_configure("contagio", foreground="#c0392b", font=("Consolas", 9, "bold"))
        self.txt_log.tag_configure("rand", foreground="#1c5cab")
        divisor.add(marco_log, weight=2)

        self._dibujar_paso_vacio()

    def _dibujar_paso_vacio(self):
        for ax in (self.ax_grid, self.ax_curva):
            ax.clear()
        self.ax_grid.set_title("Población")
        self.ax_grid.axis("off")
        self.ax_curva.set_title("Curva SEIR")
        self.canvas_paso.draw_idle()

    def _nueva_simulacion(self):
        self._detener_reproduccion()
        try:
            cfg = self.leer_configuracion()
            flujo = int(self.var_flujo.get())
        except ValueError as e:
            messagebox.showerror("Parámetros inválidos", str(e))
            return
        if cfg["archivo_semillas"]:
            semilla = None
            from pseudo_gen.utilidades.archivo_manager import ArchivoManager
            semillas = ArchivoManager.leer_semillas(cfg["archivo_semillas"])
            if flujo < len(semillas):
                semilla = semillas[flujo]
        else:
            semilla = semilla_flujo(cfg["generador"], cfg["semilla_maestra"], flujo)
        if semilla is None:
            messagebox.showerror("Flujo inválido", "No hay una semilla para ese número de flujo.")
            return

        try:
            self.rng_paso = FlujoAleatorio(crear_generador(cfg["generador"], semilla), registrar=True)
        except ValueError as e:
            messagebox.showerror("Semilla inválida", str(e))
            return
        self.cfg_paso = cfg
        self.flujo_actual = flujo
        bitacora = Bitacora(max_lineas=400)
        self.sim = SimulacionPasoAPaso(cfg, self.rng_paso, bitacora)
        self.lado = int(np.ceil(np.sqrt(cfg["poblacion"])))

        p = self.sim.params
        d_media = sum(cfg["periodo_infeccioso"]) / 2
        self.lbl_semilla.config(text=f"{GENERADORES[cfg['generador']]['nombre']} | flujo {flujo}, semilla x0 = {semilla}")
        self.lbl_params.config(
            text=(f"Parámetros muestreados:  β = {p['beta']:.4f}   p = {p['p_transmision']:.4f}   "
                  f"cobertura = {p['cobertura_vacunacion']:.4f}   efectividad = {p['efectividad_vacuna']:.4f}   "
                  f"letalidad = {p['letalidad']:.4f}      R0 ≈ β·p·D = {p['beta']:.3f}·{p['p_transmision']:.3f}·"
                  f"{d_media:g} = {p['beta'] * p['p_transmision'] * d_media:.2f}"))
        self._preparar_figura_paso()
        self._mostrar_bitacora("Inicialización (día 0)", bitacora)
        self._actualizar_paso()
        self._estado("Simulación inicializada. Use «Siguiente día» o «Reproducir».")

    def _preparar_figura_paso(self):
        cfg = self.cfg_paso
        self.ax_grid.clear()
        self.ax_curva.clear()
        self.img_grid = self.ax_grid.imshow(self._matriz_poblacion(), cmap=MAPA_COLORES, vmin=-0.5, vmax=5.5,
                                            interpolation="nearest")
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        self.ax_grid.legend(handles=[Patch(color=c, label=t) for c, t in CATEGORIAS], loc="upper center",
                            bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False)

        self.lineas = {}
        for estado in (S, E, I, R):
            (linea,) = self.ax_curva.plot([], [], color=graficas.COLOR_ESTADO[estado], linewidth=2,
                                          label=graficas.NOMBRE_LARGO[estado])
            self.lineas[estado] = linea
        self.ax_curva.set_xlim(0, cfg["dias"])
        self.ax_curva.set_ylim(0, cfg["poblacion"] * 1.02)
        self.ax_curva.set_xlabel("Día")
        self.ax_curva.set_ylabel("Personas")
        self.ax_curva.grid(True, color="#dddcd8", linewidth=0.6)
        self.ax_curva.legend(loc="center right", fontsize=8, frameon=False)
        self.ax_curva.set_title("Curva SEIR")

    def _matriz_poblacion(self):
        """Código de color de cada persona, en una matriz lado × lado."""
        pob = self.sim.pob
        estado = np.asarray(pob.estado, dtype=np.int8)
        codigo = np.where(estado == S, 0, estado + 1).astype(float)      # S→0, E→2, I→3, R→4
        vacunado = np.asarray(pob.susceptibilidad) < 1.0
        codigo[(estado == S) & vacunado] = 1
        if pob.ids_fallecidos:
            codigo[pob.ids_fallecidos] = 5
        matriz = np.full(self.lado * self.lado, np.nan)
        matriz[:len(codigo)] = codigo
        return np.ma.masked_invalid(matriz.reshape(self.lado, self.lado))

    def _actualizar_paso(self):
        sim = self.sim
        serie = np.array(sim.serie)
        dias = np.arange(len(serie))
        for estado, linea in self.lineas.items():
            linea.set_data(dias, serie[:, estado])
        self.img_grid.set_data(self._matriz_poblacion())
        c = sim.pob.conteo
        self.ax_grid.set_title(f"Población (N = {sim.pob.n:,}) - día {sim.dia}".replace(",", "."))
        self.canvas_paso.draw_idle()

        estado_brote = ""
        if sim.extinguida:
            estado_brote = "   |   Brote extinguido"
        elif sim.terminada:
            estado_brote = "   |   Fin del horizonte"
        self.lbl_conteo.config(
            text=(f"Día {sim.dia}/{self.cfg_paso['dias']}    S = {c[S]:,}   E = {c[E]:,}   I = {c[I]:,}   "
                  f"R = {c[R]:,}   (fallecidos {sim.pob.fallecidos:,})    vacunados = {sim.pob.vacunados:,}    "
                  f"infectados acumulados = {sim.pob.total_infectados:,}    R_i usados = {self.rng_paso.usados:,}"
                  f"{estado_brote}").replace(",", "."))

    def _mostrar_bitacora(self, titulo, bitacora):
        self.txt_log.delete("1.0", "end")
        self.txt_log.insert("end", f"{titulo}\n", "dia")
        for linea in bitacora.lineas:
            if linea.startswith("\n══"):
                continue
            tag = "contagio" if "CONTAGIO" in linea or "fallece" in linea else ("rand" if "R=" in linea else "")
            self.txt_log.insert("end", linea + "\n", tag)
        if bitacora.omitidas:
            self.txt_log.insert("end", f"... ({bitacora.omitidas:,} líneas más omitidas)\n".replace(",", "."))

    def _siguiente_dia(self):
        if self.sim is None:
            self._nueva_simulacion()
            return False
        if self.sim.terminada or self.sim.extinguida:
            self._detener_reproduccion()
            return False
        usados_antes = self.rng_paso.usados
        bitacora = Bitacora(max_lineas=400)
        self.sim.avanzar(bitacora)
        self._mostrar_bitacora(f"Día {self.sim.dia}: {self.rng_paso.usados - usados_antes:,} números "
                               f"pseudoaleatorios usados".replace(",", "."), bitacora)
        self._actualizar_paso()
        return True

    def _alternar_reproduccion(self):
        if self.reproduciendo:
            self._detener_reproduccion()
        else:
            if self.sim is None:
                self._nueva_simulacion()
            self.reproduciendo = True
            self.btn_reproducir.config(text="⏸ Pausar")
            self._paso_reproduccion()

    def _paso_reproduccion(self):
        if not self.reproduciendo:
            return
        if self._siguiente_dia():
            self.after(int(1000 / max(1.0, self.var_velocidad.get())), self._paso_reproduccion)
        else:
            self._detener_reproduccion()

    def _detener_reproduccion(self):
        self.reproduciendo = False
        self.btn_reproducir.config(text="▶ Reproducir")

    def _ir_al_final(self):
        self._detener_reproduccion()
        if self.sim is None:
            self._nueva_simulacion()
        desde = self.sim.dia + 1
        while not self.sim.terminada and not self.sim.extinguida:
            self.sim.avanzar()
        if self.sim.dia >= desde:
            self.txt_log.delete("1.0", "end")
            self.txt_log.insert("end", f"Días {desde}–{self.sim.dia} simulados sin detalle "
                                       f"(use «Siguiente día» para ver la bitácora de un día).\n", "dia")
        self._actualizar_paso()

    # ── Pestaña: Montecarlo ─────────────────────────────────────────────

    def _construir_montecarlo(self):
        pestana = ttk.Frame(self.pestanas, padding=6)
        self.pestanas.add(pestana, text="Montecarlo")

        controles = ttk.Frame(pestana)
        controles.pack(fill="x")
        ttk.Label(controles, text="Simulaciones").pack(side="left")
        self.var_n = tk.StringVar(value="200")
        ttk.Spinbox(controles, from_=10, to=2000, increment=10, textvariable=self.var_n, width=7
                    ).pack(side="left", padx=4)
        ttk.Label(controles, text="Procesos").pack(side="left", padx=(8, 0))
        self.var_procesos = tk.StringVar(value=str(os.cpu_count() or 1))
        ttk.Spinbox(controles, from_=1, to=os.cpu_count() or 1, textvariable=self.var_procesos, width=4
                    ).pack(side="left", padx=4)
        self.var_validar = tk.BooleanVar(value=True)
        ttk.Checkbutton(controles, text="Validar R_i de cada simulación", variable=self.var_validar
                        ).pack(side="left", padx=8)
        self.btn_escenario = ttk.Button(controles, text="Ejecutar escenario", command=self._ejecutar_escenario)
        self.btn_escenario.pack(side="left", padx=4)
        self.btn_comparar = ttk.Button(controles, text="Comparar sin vacunación vs. escenario",
                                       command=self._ejecutar_comparacion)
        self.btn_comparar.pack(side="left", padx=4)
        ttk.Button(controles, text="Guardar resultados…", command=self._guardar_resultados).pack(side="left", padx=4)

        avance = ttk.Frame(pestana)
        avance.pack(fill="x", pady=4)
        self.barra_mc = ttk.Progressbar(avance, maximum=100, length=300)
        self.barra_mc.pack(side="left")
        self.lbl_mc = ttk.Label(avance, text="")
        self.lbl_mc.pack(side="left", padx=8)

        selector = ttk.Frame(pestana)
        selector.pack(fill="x")
        ttk.Label(selector, text="Escenario").pack(side="left")
        self.var_escenario = tk.StringVar()
        self.cmb_escenario = ttk.Combobox(selector, textvariable=self.var_escenario, state="readonly", width=24)
        self.cmb_escenario.pack(side="left", padx=4)
        ttk.Label(selector, text="Gráfica").pack(side="left", padx=(10, 0))
        self.var_grafica = tk.StringVar(value=GRAFICAS_ESCENARIO[0])
        self.cmb_grafica = ttk.Combobox(selector, textvariable=self.var_grafica, state="readonly", width=34,
                                        values=GRAFICAS_ESCENARIO)
        self.cmb_grafica.pack(side="left", padx=4)
        self.cmb_escenario.bind("<<ComboboxSelected>>", lambda _e: self._mostrar_grafica_mc())
        self.cmb_grafica.bind("<<ComboboxSelected>>", lambda _e: self._mostrar_grafica_mc())
        self.lbl_resumen_mc = ttk.Label(selector, text="", foreground="#52514e")
        self.lbl_resumen_mc.pack(side="left", padx=10)

        divisor = ttk.Panedwindow(pestana, orient="vertical")
        divisor.pack(fill="both", expand=True, pady=(4, 0))
        self.panel_mc = PanelFigura(divisor)
        divisor.add(self.panel_mc, weight=3)
        columnas = ("Escenario", "Métrica", "n", "Media", "Desv.", "IC 95 % media", "P2.5", "Mediana", "P97.5")
        marco, self.tabla_mc = crear_tabla(divisor, columnas, (130, 190, 60, 90, 90, 170, 80, 80, 80), alto=7)
        divisor.add(marco, weight=1)

    def _parametros_mc(self):
        cfg = self.leer_configuracion(validar_numeros="si" if self.var_validar.get() else "no")
        n = int(self.var_n.get())
        procesos = int(self.var_procesos.get())
        if n < 2 or procesos < 1:
            raise ValueError("Se requieren al menos 2 simulaciones y 1 proceso.")
        return cfg, n, procesos

    def _ejecutar_escenario(self):
        try:
            cfg, n, procesos = self._parametros_mc()
        except ValueError as e:
            messagebox.showerror("Parámetros inválidos", str(e))
            return
        self._lanzar_montecarlo([cfg], n, procesos)

    def _ejecutar_comparacion(self):
        try:
            cfg, n, procesos = self._parametros_mc()
        except ValueError as e:
            messagebox.showerror("Parámetros inválidos", str(e))
            return
        if cfg["cobertura_vacunacion"] == (0.0, 0.0):
            messagebox.showinfo("Comparación", "El escenario actual ya tiene cobertura 0. "
                                               "Escriba una cobertura mayor que 0 para comparar.")
            return
        sin = cfg.copia(cobertura_vacunacion=(0, 0), nombre="sin_vacunacion")
        if cfg["nombre"] == "sin_vacunacion":
            cfg = cfg.copia(nombre="con_vacunacion")
        self._lanzar_montecarlo([sin, cfg], n, procesos)

    def _lanzar_montecarlo(self, configuraciones, n, procesos):
        if self.trabajando:
            return
        self._bloquear(True)
        self.barra_mc["value"] = 0
        total = len(configuraciones)

        def trabajo():
            resultados = []
            for k, cfg in enumerate(configuraciones):
                def progreso(aceptadas, rechazadas, objetivo, k=k, nombre=cfg["nombre"]):
                    self.cola.put(("progreso_mc", 100 * (k + aceptadas / objetivo) / total,
                                   f"{nombre}: {aceptadas}/{objetivo} aceptadas, {rechazadas} rechazadas"))
                resultados.append(ejecutar_montecarlo(cfg, n, procesos, mostrar_progreso=False, progreso=progreso))
            return resultados

        self._en_hilo(trabajo, "fin_mc")

    def _fin_montecarlo(self, resultados):
        self.resultados = {r.nombre: r for r in resultados}
        nombres = list(self.resultados)
        opciones = GRAFICAS_ESCENARIO + (GRAFICAS_COMPARACION if len(nombres) > 1 else [])
        self.cmb_escenario["values"] = nombres
        self.var_escenario.set(nombres[-1])
        self.cmb_grafica["values"] = opciones
        self.var_grafica.set(GRAFICAS_COMPARACION[0] if len(nombres) > 1 else GRAFICAS_ESCENARIO[0])

        for fila in self.tabla_mc.get_children():
            self.tabla_mc.delete(fila)
        for r in resultados:
            for f in analisis.estadisticas_descriptivas(r):
                if not f["n_validos"]:
                    continue
                self.tabla_mc.insert("", "end", values=(
                    f["escenario"], f["metrica"], f["n_validos"], formato_numero(f["media"]),
                    formato_numero(f["desviacion"]),
                    f"[{formato_numero(f['ic95_inf'])}, {formato_numero(f['ic95_sup'])}]",
                    formato_numero(f["p2_5"]), formato_numero(f["mediana"]), formato_numero(f["p97_5"])))
        self.barra_mc["value"] = 100
        partes = [f"{r.nombre}: {len(r.aceptadas)} aceptadas / {len(r.rechazadas)} rechazadas, "
                  f"{r.tiempo_total:.1f} s" for r in resultados]
        self.lbl_mc.config(text="   |   ".join(partes))
        self._mostrar_grafica_mc()

    def _figura_mc(self, grafica, res):
        control = res.columna("dia_control")
        sin_control = int(np.isnan(control).sum())
        figuras = {
            "Curva epidémica": lambda: graficas.curva_epidemica(res),
            "Magnitud del pico": lambda: graficas.histograma(res.columna("pico_infectados"),
                                                             f"Magnitud del pico - {res.nombre}", "Máximo de I(t)"),
            "Día del pico": lambda: graficas.histograma(res.columna("dia_pico"), f"Día del pico - {res.nombre}",
                                                        "Día en que I(t) es máximo"),
            "Total de infectados": lambda: graficas.histograma(res.columna("total_infectados"),
                                                               f"Total de infectados - {res.nombre}", "Personas"),
            "Fallecidos": lambda: graficas.histograma(res.columna("fallecidos"), f"Fallecidos - {res.nombre}",
                                                      "Personas"),
            "Días hasta control": lambda: graficas.histograma(
                control, f"Días hasta el control - {res.nombre}",
                f"Primer día después del pico con E + I ≤ {res.cfg['umbral_control']}",
                nota=f"{sin_control} de {len(control)} brotes sin control en {res.cfg['dias']} días (excluidos)"
                if sin_control else None),
            "Parámetros muestreados": lambda: graficas.parametros_muestreados(res),
            "Validación de los R_i": lambda: graficas.validacion_pruebas(res),
            "Sensibilidad global (Spearman)": lambda: graficas.correlaciones(analisis.correlaciones_spearman(res)),
            "Comparación: curvas": lambda: graficas.comparar_curvas(list(self.resultados.values())),
            "Comparación: distribuciones": lambda: graficas.comparar_distribuciones(list(self.resultados.values())),
        }
        return figuras[grafica]()

    def _mostrar_grafica_mc(self):
        res = self.resultados.get(self.var_escenario.get())
        if res is None:
            return
        n = len(res.aceptadas)
        tasa = res.columna("tasa_ataque")
        self.lbl_resumen_mc.config(text=f"{res.nombre}: tasa de ataque media {100 * tasa.mean():.1f} %, "
                                        f"{n} simulaciones")
        fig = self._figura_mc(self.var_grafica.get(), res)
        if fig is None:
            messagebox.showinfo("Sin datos", "Esta gráfica no tiene datos para el escenario "
                                             "(p. ej., validación desactivada o parámetros fijos).")
        self.panel_mc.mostrar(fig)

    def _guardar_resultados(self):
        if not self.resultados:
            messagebox.showinfo("Guardar", "Primero ejecute un escenario.")
            return
        carpeta = filedialog.askdirectory(initialdir=DIR_BASE, title="Carpeta donde guardar los resultados")
        if not carpeta:
            return
        for r in self.resultados.values():
            destino = os.path.join(carpeta, r.nombre)
            r.guardar(destino)
            analisis.guardar_tabla(analisis.estadisticas_descriptivas(r), os.path.join(destino, "estadisticas.csv"))
            for grafica in GRAFICAS_ESCENARIO:
                fig = self._figura_mc(grafica, r)
                if fig is not None:
                    nombre = grafica.lower().replace(" ", "_").replace("(", "").replace(")", "")
                    fig.savefig(os.path.join(destino, f"{nombre}.png"), dpi=200)
                    plt.close(fig)
        if len(self.resultados) > 1:
            for grafica in GRAFICAS_COMPARACION:
                fig = self._figura_mc(grafica, next(iter(self.resultados.values())))
                nombre = grafica.lower().replace(": ", "_").replace(" ", "_")
                fig.savefig(os.path.join(carpeta, f"{nombre}.png"), dpi=200)
                plt.close(fig)
        self._estado(f"Resultados guardados en {carpeta}")

    # ── Pestaña: Sensibilidad ───────────────────────────────────────────

    def _construir_sensibilidad(self):
        pestana = ttk.Frame(self.pestanas, padding=6)
        self.pestanas.add(pestana, text="Sensibilidad")
        ttk.Label(pestana, wraplength=1100, foreground="#52514e", text=(
            "Cada parámetro se fija en su valor mínimo y luego en su máximo, mientras los demás se siguen "
            "muestreando de sus distribuciones. Todas las corridas usan los mismos flujos de números "
            "(números aleatorios comunes), así que la diferencia se debe al parámetro y no al azar. "
            "La barra más larga es el parámetro con más impacto.")).pack(anchor="w")

        controles = ttk.Frame(pestana)
        controles.pack(fill="x", pady=6)
        ttk.Label(controles, text="Simulaciones por caso").pack(side="left")
        self.var_n_sens = tk.StringVar(value="50")
        ttk.Spinbox(controles, from_=10, to=1000, increment=10, textvariable=self.var_n_sens, width=6
                    ).pack(side="left", padx=4)
        self.btn_sens = ttk.Button(controles, text="Ejecutar análisis", command=self._ejecutar_sensibilidad)
        self.btn_sens.pack(side="left", padx=6)
        self.barra_sens = ttk.Progressbar(controles, maximum=100, length=250)
        self.barra_sens.pack(side="left", padx=6)
        self.lbl_sens = ttk.Label(controles, text="")
        self.lbl_sens.pack(side="left", padx=6)

        divisor = ttk.Panedwindow(pestana, orient="vertical")
        divisor.pack(fill="both", expand=True)
        self.panel_sens = PanelFigura(divisor)
        divisor.add(self.panel_sens, weight=3)
        columnas = ("Parámetro", "Mínimo", "Máximo", "Tasa ataque (mín)", "Tasa ataque (máx)",
                    "Amplitud (p. p.)", "Pico (mín)", "Pico (máx)")
        marco, self.tabla_sens = crear_tabla(divisor, columnas, (190, 70, 70, 120, 120, 110, 90, 90), alto=6)
        divisor.add(marco, weight=1)

    def _ejecutar_sensibilidad(self):
        if self.trabajando:
            return
        try:
            cfg = self.leer_configuracion()
            n = int(self.var_n_sens.get())
            procesos = int(self.var_procesos.get())
        except ValueError as e:
            messagebox.showerror("Parámetros inválidos", str(e))
            return
        self._bloquear(True)
        self.barra_sens["value"] = 0

        def progreso(caso, total, descripcion):
            self.cola.put(("progreso_sens", 100 * caso / total, f"Caso {min(caso + 1, total)}/{total}: {descripcion}"))

        self._en_hilo(lambda: analisis.sensibilidad_oat(cfg, n, procesos, progreso=progreso), "fin_sens")

    def _fin_sensibilidad(self, sens):
        self.sensibilidad = sens
        for fila in self.tabla_sens.get_children():
            self.tabla_sens.delete(fila)
        filas = sorted(sens["parametros"], key=lambda f: abs(f[4]["tasa_ataque"] - f[3]["tasa_ataque"]),
                       reverse=True)
        for p, a, b, bajo, alto in filas:
            self.tabla_sens.insert("", "end", values=(
                analisis.ETIQUETAS_PARAMETROS[p], f"{a:g}", f"{b:g}",
                f"{100 * bajo['tasa_ataque']:.1f} %", f"{100 * alto['tasa_ataque']:.1f} %",
                f"{100 * abs(alto['tasa_ataque'] - bajo['tasa_ataque']):.1f}",
                f"{bajo['pico_infectados']:.0f}", f"{alto['pico_infectados']:.0f}"))
        self.barra_sens["value"] = 100
        self.lbl_sens.config(text=f"Terminado. Caso base: tasa de ataque {100 * sens['base']['tasa_ataque']:.1f} %")
        self.panel_sens.mostrar(graficas.tornado(sens))

    # ── Pestaña: Números pseudoaleatorios ───────────────────────────────

    def _construir_numeros(self):
        pestana = ttk.Frame(self.pestanas, padding=6)
        self.pestanas.add(pestana, text="Números pseudoaleatorios")
        ttk.Label(pestana, wraplength=1100, foreground="#52514e", text=(
            "Todos los números vienen del módulo pseudo_gen. Aquí se puede comprobar que los R_i que usó la "
            "simulación de la pestaña «Paso a paso» pasan las 5 pruebas, y que los flujos de simulaciones "
            "distintas no están correlacionados entre sí.")).pack(anchor="w")

        controles = ttk.Frame(pestana)
        controles.pack(fill="x", pady=6)
        ttk.Button(controles, text="Validar los R_i usados en «Paso a paso»",
                   command=self._validar_paso).pack(side="left")
        self.btn_flujos = ttk.Button(controles, text="Verificar independencia entre flujos",
                                     command=self._verificar_flujos)
        self.btn_flujos.pack(side="left", padx=6)
        self.lbl_numeros = ttk.Label(controles, text="")
        self.lbl_numeros.pack(side="left", padx=6)

        divisor = ttk.Panedwindow(pestana, orient="horizontal")
        divisor.pack(fill="both", expand=True)
        izquierda = ttk.Frame(divisor)
        divisor.add(izquierda, weight=1)
        marco, self.tabla_pruebas = crear_tabla(izquierda, ("Prueba", "Estadístico", "Resultado"),
                                                (150, 110, 100), alto=7)
        marco.pack(fill="x")
        ttk.Label(izquierda, text="Primeros R_i usados", style="Titulo.TLabel").pack(anchor="w", pady=(10, 0))
        marco, self.tabla_ri = crear_tabla(izquierda, ("i", "R_i"), (60, 120), alto=14)
        marco.pack(fill="both", expand=True)
        self.panel_numeros = PanelFigura(divisor)
        divisor.add(self.panel_numeros, weight=3)

    def _llenar_pruebas(self, resultados):
        for fila in self.tabla_pruebas.get_children():
            self.tabla_pruebas.delete(fila)
        for nombre, ok, est in resultados:
            self.tabla_pruebas.insert("", "end", values=(nombre, f"{est:.6f}", "APROBADA" if ok else "RECHAZADA"))

    def _validar_paso(self):
        if self.sim is None:
            messagebox.showinfo("Validar", "Primero cree una simulación en la pestaña «Paso a paso».")
            return
        ri = self.rng_paso.registro
        if len(ri) < 100:
            messagebox.showinfo("Validar", "Hay muy pocos números para validar.")
            return
        aprobada, resultados = validar_secuencia(ri, self.cfg_paso["alpha"])
        self._llenar_pruebas(resultados)
        for fila in self.tabla_ri.get_children():
            self.tabla_ri.delete(fila)
        for k, r in enumerate(ri[:200], start=1):
            self.tabla_ri.insert("", "end", values=(k, f"{r:.6f}"))
        self.lbl_numeros.config(text=(f"Flujo {self.flujo_actual}, hasta el día {self.sim.dia}: "
                                      f"{len(ri):,} R_i → ".replace(",", ".")
                                      + ("aprueba las 5 pruebas" if aprobada else
                                         "NO aprueba todas las pruebas: en Montecarlo esta simulación se descartaría")))
        fig = graficas.histograma(ri, f"R_i usados por la simulación (flujo {self.flujo_actual})", "R_i",
                                  nota=f"{len(ri):,} números; se espera una distribución plana en [0, 1)",
                                  etiqueta_y="Frecuencia")
        self.panel_numeros.mostrar(fig)

    def _verificar_flujos(self):
        if self.trabajando:
            return
        try:
            cfg = self.leer_configuracion()
        except ValueError as e:
            messagebox.showerror("Parámetros inválidos", str(e))
            return
        self._bloquear(True)
        self.lbl_numeros.config(text="Calculando el primer R_i de cada flujo...")

        def trabajo():
            tipo, maestra = cfg["generador"], cfg["semilla_maestra"]
            p = GENERADORES[tipo]
            from aleatorio import max_flujos
            n = max_flujos(tipo)
            aprobada, resultados, rho, primeros = validar_independencia_flujos(tipo, maestra, n, cfg["alpha"])
            malos = None
            if tipo == "lcg":
                malos = [crear_generador(tipo, saltar(maestra, k * 2**21, p["a"], p["c"], p["m"])).siguiente()
                         for k in range(n)]
            return tipo, n, aprobada, resultados, rho, primeros, malos

        self._en_hilo(trabajo, "fin_flujos")

    def _fin_flujos(self, datos):
        tipo, n, aprobada, resultados, rho, primeros, malos = datos
        self._llenar_pruebas(resultados)
        for fila in self.tabla_ri.get_children():
            self.tabla_ri.delete(fila)
        for k, r in enumerate(primeros[:200]):
            self.tabla_ri.insert("", "end", values=(f"flujo {k}", f"{r:.6f}"))
        self.lbl_numeros.config(text=f"Primer R_i de {n} flujos: "
                                     + ("aprueba las 5 pruebas" if aprobada else "NO aprueba todas")
                                     + f"; correlación entre flujos consecutivos = {rho:+.4f}")
        salto = GENERADORES[tipo]["salto"]
        if malos is not None:
            fig = graficas.dispersion_flujos(malos, primeros, f"Salto 2^21 = {2**21:,} (descartado)",
                                             f"Salto 2^21 + 1 = {salto:,} (usado)")
        else:
            fig = graficas.dispersion_flujos(primeros, primeros, f"Salto {salto:,}", f"Salto {salto:,}")
        self.panel_numeros.mostrar(fig)

    # ── Pestaña: ayuda ──────────────────────────────────────────────────

    def _construir_ayuda(self):
        pestana = ttk.Frame(self.pestanas, padding=6)
        self.pestanas.add(pestana, text="¿Cómo funciona?")
        texto = tk.Text(pestana, wrap="word", font=("Consolas", 10), padx=10, pady=10)
        barra = ttk.Scrollbar(pestana, orient="vertical", command=texto.yview)
        texto.configure(yscrollcommand=barra.set)
        texto.insert("1.0", TEXTO_AYUDA)
        texto.configure(state="disabled")
        texto.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")

    # ── Hilos de trabajo ────────────────────────────────────────────────

    def _en_hilo(self, funcion, mensaje_fin):
        """Ejecuta funcion() en segundo plano; su resultado llega por la cola como (mensaje_fin, resultado)."""
        def envoltura():
            try:
                self.cola.put((mensaje_fin, funcion()))
            except Exception as e:     # Se muestra en la interfaz en lugar de perderse en el hilo
                traceback.print_exc()
                self.cola.put(("error", f"{type(e).__name__}: {e}"))
        threading.Thread(target=envoltura, daemon=True).start()

    def _revisar_cola(self):
        try:
            while True:
                mensaje = self.cola.get_nowait()
                tipo = mensaje[0]
                if tipo == "progreso_mc":
                    self.barra_mc["value"] = mensaje[1]
                    self.lbl_mc.config(text=mensaje[2])
                elif tipo == "progreso_sens":
                    self.barra_sens["value"] = mensaje[1]
                    self.lbl_sens.config(text=mensaje[2])
                else:
                    self._bloquear(False)
                    if tipo == "fin_mc":
                        self._fin_montecarlo(mensaje[1])
                    elif tipo == "fin_sens":
                        self._fin_sensibilidad(mensaje[1])
                    elif tipo == "fin_flujos":
                        self._fin_flujos(mensaje[1])
                    elif tipo == "error":
                        messagebox.showerror("Error", mensaje[1])
        except queue.Empty:
            pass
        self.after(100, self._revisar_cola)

    def _bloquear(self, ocupado):
        self.trabajando = ocupado
        estado = "disabled" if ocupado else "normal"
        for boton in (self.btn_escenario, self.btn_comparar, self.btn_sens, self.btn_flujos):
            boton.config(state=estado)
        self._estado("Trabajando..." if ocupado else "Listo.")

    def _estado(self, texto):
        self.barra_estado.config(text=texto)


if __name__ == "__main__":
    # Los módulos de análisis imprimen caracteres como «→»; la consola de Windows puede no soportarlos.
    import sys
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    Aplicacion().mainloop()
