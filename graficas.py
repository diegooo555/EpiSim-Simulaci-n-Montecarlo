"""
Gráficas de EpiSim (Matplotlib). Todas se guardan en PNG a 200 dpi.

Colores fijos por entidad (no por posición): S, E, I, R siempre con el mismo color, y cada
escenario siempre con el mismo color en todas las gráficas comparativas.
"""
import math
import os

import matplotlib

matplotlib.use("Agg")   # Solo se guardan archivos; no se abren ventanas
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analisis import ETIQUETAS_PARAMETROS
from modelo import E, I, R, S

COLOR_ESTADO = {S: "#2a78d6", E: "#eb6834", I: "#1baf7a", R: "#c98500"}
NOMBRE_LARGO = {S: "Susceptibles (S)", E: "Expuestos (E)", I: "Infecciosos (I)", R: "Recuperados/removidos (R)"}
COLORES_ESCENARIO = ["#2a78d6", "#eb6834", "#1baf7a", "#c98500"]
COLOR_BAJO, COLOR_ALTO = "#2a78d6", "#e34948"
TINTA, TINTA_SEC, REJILLA = "#0b0b0b", "#52514e", "#dddcd8"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": TINTA_SEC,
    "axes.labelcolor": TINTA,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.6,
    "xtick.color": TINTA_SEC,
    "ytick.color": TINTA_SEC,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def _guardar(fig, ruta):
    """Guarda la figura en `ruta` y la cierra; si ruta es None la devuelve (interfaz gráfica)."""
    fig.tight_layout()
    if ruta is None:
        return fig
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"  Gráfica: {ruta}")


def _bandas(series):
    """Media, IC 95 % de la media y banda del 95 % de las trayectorias (P2.5–P97.5)."""
    n = series.shape[0]
    media = series.mean(axis=0)
    if n > 1:
        mitad = stats.t.ppf(0.975, n - 1) * series.std(axis=0, ddof=1) / math.sqrt(n)
    else:
        mitad = np.zeros_like(media)
    p_inf, p_sup = np.percentile(series, [2.5, 97.5], axis=0)
    return media, media - mitad, media + mitad, p_inf, p_sup


# ── Curva epidémica ─────────────────────────────────────────────────────

def curva_epidemica(res, ruta=None):
    """
    Curvas S, E, I, R: línea = promedio; banda oscura = IC 95 % del promedio;
    banda clara = 95 % central de las simulaciones (P2.5–P97.5).
    """
    series = res.series
    n_sim, n_dias, _ = series.shape
    dias = np.arange(n_dias)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    for estado, ax_lista in ((S, [ax1]), (E, [ax1, ax2]), (I, [ax1, ax2]), (R, [ax1])):
        media, ic_inf, ic_sup, p_inf, p_sup = _bandas(series[:, :, estado])
        c = COLOR_ESTADO[estado]
        for ax in ax_lista:
            ax.fill_between(dias, p_inf, p_sup, color=c, alpha=0.13, linewidth=0)
            ax.fill_between(dias, ic_inf, ic_sup, color=c, alpha=0.35, linewidth=0)
            ax.plot(dias, media, color=c, label=NOMBRE_LARGO[estado])

    ax1.set_title(f"Curva epidémica SEIR - {res.nombre} ({n_sim} simulaciones)")
    ax1.set_ylabel("Personas")
    ax2.set_title("Detalle de E e I")
    for ax in (ax1, ax2):
        ax.set_xlabel("Día")
        ax.set_xlim(0, n_dias - 1)
        ax.set_ylim(bottom=0)
    ax1.legend(loc="center right")
    ax2.text(0.99, 0.97, "Línea: promedio\nBanda oscura: IC 95 % del promedio\n"
             "Banda clara: 95 % de las simulaciones (P2.5–P97.5)",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8.5, color=TINTA_SEC)
    return _guardar(fig, ruta)


# ── Histogramas ─────────────────────────────────────────────────────────

def histograma(valores, titulo, etiqueta_x, ruta=None, color=COLORES_ESCENARIO[0], nota=None,
               etiqueta_y="Frecuencia (simulaciones)"):
    """Histograma de una métrica con líneas de media y mediana."""
    x = np.asarray(valores, dtype=float)
    x = x[~np.isnan(x)]
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(x) == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
    else:
        enteros = np.all(x == np.round(x)) and np.ptp(x) < 60
        bins = np.arange(x.min() - 0.5, x.max() + 1.5) if enteros else "fd"
        ax.hist(x, bins=bins, color=color, edgecolor="white", linewidth=0.8, alpha=0.9)
        ax.axvline(x.mean(), color=TINTA, linestyle="--", linewidth=1.3, label=f"Media = {x.mean():,.1f}")
        ax.axvline(np.median(x), color=TINTA_SEC, linestyle=":", linewidth=1.5,
                   label=f"Mediana = {np.median(x):,.1f}")
        ax.legend(loc="best")
    ax.set_title(titulo)
    ax.set_xlabel(etiqueta_x + (f"\n{nota}" if nota else ""))
    ax.set_ylabel(etiqueta_y)
    return _guardar(fig, ruta)


def histogramas_escenario(res, carpeta):
    """Distribución de picos (magnitud y día), total de infectados, fallecidos y días hasta control."""
    n = len(res.aceptadas)
    control = res.columna("dia_control")
    sin_control = int(np.isnan(control).sum())
    especificos = [
        ("pico_infectados", "Magnitud del pico de infecciosos", "Máximo de I(t) (personas)", None),
        ("dia_pico", "Día del pico de infecciosos", "Día en que I(t) es máximo", None),
        ("total_infectados", "Total de infectados por simulación", "Personas infectadas (incluye casos índice)", None),
        ("fallecidos", "Total de fallecidos por simulación", "Fallecidos", None),
        ("dia_control", "Días hasta el control del brote",
         f"Primer día después del pico con E + I ≤ {res.cfg['umbral_control']}",
         f"{sin_control} de {n} brotes sin control en {res.cfg['dias']} días (excluidos)" if sin_control else None),
    ]
    for clave, titulo, etiqueta, nota in especificos:
        histograma(res.columna(clave), f"{titulo} - {res.nombre}", etiqueta,
                   os.path.join(carpeta, f"hist_{clave}.png"), nota=nota)


def parametros_muestreados(res, ruta=None):
    """Histogramas de los parámetros muestreados con U(a, b): evidencia del uso del generador."""
    claves = [p for p in ("beta", "p_transmision", "cobertura_vacunacion", "efectividad_vacuna", "letalidad")
              if np.ptp(res.columna(p)) > 0]
    fig, ejes = plt.subplots(1, len(claves), figsize=(3.3 * len(claves), 3.6), squeeze=False)
    for ax, clave in zip(ejes[0], claves):
        x = res.columna(clave)
        a, b = res.cfg[clave]
        ax.hist(x, bins=10, range=(a, b), color=COLORES_ESCENARIO[0], edgecolor="white")
        ax.axhline(len(x) / 10, color=TINTA, linestyle="--", linewidth=1.2, label="Esperado U(a, b)")
        ax.set_title(f"{ETIQUETAS_PARAMETROS[clave]}\nU({a}, {b})", fontsize=9.5)
        ax.set_xlabel("Valor muestreado")
    ejes[0][0].set_ylabel("Frecuencia")
    ejes[0][0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Parámetros muestreados por simulación - {res.nombre}", fontweight="bold")
    return _guardar(fig, ruta)


# ── Validación de los números pseudoaleatorios ──────────────────────────

def validacion_pruebas(res, ruta=None):
    """Porcentaje de simulaciones (aceptadas + rechazadas) cuyos R_i aprobaron cada prueba."""
    todas = [s for s in res.aceptadas + res.rechazadas if "validacion" in s]
    if not todas:
        return
    nombres = [n for n, _, _ in todas[0]["validacion"]]
    tasas = [100 * np.mean([s["validacion"][k][1] for s in todas]) for k in range(len(nombres))]
    alpha = res.cfg["alpha"]

    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(nombres, tasas, color=COLORES_ESCENARIO[0], width=0.6)
    ax.axhline(100 * (1 - alpha), color=TINTA, linestyle="--", linewidth=1.3,
               label=f"Esperado si los R_i son U(0,1) independientes: {100 * (1 - alpha):.0f} %")
    for b, t in zip(barras, tasas):
        ax.annotate(f"{t:.1f} %", (b.get_x() + b.get_width() / 2, t), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Simulaciones que aprueban (%)")
    ax.set_title(f"Validación de los R_i usados en cada simulación - {res.nombre} "
                 f"({len(todas)} simulaciones, α = {alpha})")
    ax.legend(loc="lower right")
    return _guardar(fig, ruta)


def dispersion_flujos(primeros_malos, primeros_buenos, etiqueta_mala, etiqueta_buena, ruta=None):
    """Gráfico de retardo (R de flujo k vs flujo k+1): muestra la estructura de un salto mal elegido."""
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, datos, titulo, c in ((ejes[0], primeros_malos, etiqueta_mala, COLOR_ALTO),
                                 (ejes[1], primeros_buenos, etiqueta_buena, COLORES_ESCENARIO[0])):
        x = np.asarray(datos)
        rho = np.corrcoef(x[:-1], x[1:])[0, 1]
        ax.scatter(x[:-1], x[1:], s=8, color=c, alpha=0.7, linewidths=0)
        ax.set_title(f"{titulo}\ncorrelación entre flujos consecutivos = {rho:.3f}", fontsize=10)
        ax.set_xlabel("Primer R_i del flujo k")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    ejes[0].set_ylabel("Primer R_i del flujo k + 1")
    return _guardar(fig, ruta)


# ── Comparación de escenarios ───────────────────────────────────────────

def comparar_curvas(resultados, ruta=None):
    """I(t) y acumulado de infectados (N − S(t)) de cada escenario, con bandas del 95 %."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for k, res in enumerate(resultados):
        c = COLORES_ESCENARIO[k % len(COLORES_ESCENARIO)]
        dias = np.arange(res.series.shape[1])
        acumulado = res.cfg["poblacion"] - res.series[:, :, S]
        for ax, datos in ((ax1, res.series[:, :, I]), (ax2, acumulado)):
            media, ic_inf, ic_sup, p_inf, p_sup = _bandas(datos)
            ax.fill_between(dias, p_inf, p_sup, color=c, alpha=0.13, linewidth=0)
            ax.fill_between(dias, ic_inf, ic_sup, color=c, alpha=0.35, linewidth=0)
            ax.plot(dias, media, color=c, label=res.nombre)
    ax1.set_title("Infecciosos activos I(t)")
    ax1.set_ylabel("Personas")
    ax2.set_title("Infectados acumulados (N − S(t))")
    for ax in (ax1, ax2):
        ax.set_xlabel("Día")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right" if ax is ax1 else "lower right")
    fig.suptitle("Comparación de escenarios (línea: promedio; bandas: IC 95 % del promedio y P2.5–P97.5)",
                 fontweight="bold")
    return _guardar(fig, ruta)


def comparar_distribuciones(resultados, ruta=None):
    """Diagramas de caja de tasa de ataque, pico y fallecidos por escenario."""
    metricas = [("tasa_ataque", "Tasa de ataque", 100, "% de la población"),
                ("pico_infectados", "Pico de infecciosos", 1, "Personas"),
                ("fallecidos", "Fallecidos", 1, "Personas")]
    fig, ejes = plt.subplots(1, 3, figsize=(14, 5))
    nombres = [r.nombre for r in resultados]
    for ax, (clave, titulo, escala, unidad) in zip(ejes, metricas):
        datos = [r.columna(clave) * escala for r in resultados]
        caja = ax.boxplot(datos, patch_artist=True, widths=0.5, showmeans=True,
                          medianprops={"color": TINTA, "linewidth": 1.5},
                          meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": TINTA,
                                     "markersize": 6},
                          flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})
        for k, parche in enumerate(caja["boxes"]):
            parche.set_facecolor(COLORES_ESCENARIO[k % len(COLORES_ESCENARIO)])
            parche.set_alpha(0.75)
        ax.set_xticks(range(1, len(nombres) + 1), nombres)
        ax.set_title(titulo)
        ax.set_ylabel(unidad)
    fig.suptitle("Distribución de resultados por escenario (◇ = media)", fontweight="bold")
    return _guardar(fig, ruta)


# ── Sensibilidad ────────────────────────────────────────────────────────

def tornado(sens, ruta=None):
    """Un panel por métrica; barras = cambio de la media al llevar el parámetro a su mínimo o máximo."""
    etiquetas_metricas = {"tasa_ataque": ("Tasa de ataque", 100, "puntos porcentuales"),
                          "pico_infectados": ("Pico de infecciosos", 1, "personas"),
                          "fallecidos": ("Fallecidos", 1, "personas")}
    metricas = sens["metricas"]
    fig, ejes = plt.subplots(1, len(metricas), figsize=(6.2 * len(metricas), 0.7 * len(sens["parametros"]) + 2.2),
                             squeeze=False)
    for ax, m in zip(ejes[0], metricas):
        titulo, escala, unidad = etiquetas_metricas.get(m, (m, 1, ""))
        base = sens["base"][m] * escala
        filas = sorted(sens["parametros"], key=lambda f: abs(f[4][m] - f[3][m]))   # Mayor arriba
        y = np.arange(len(filas))
        for k, (p, a, b, bajo, alto) in enumerate(filas):
            d_bajo, d_alto = bajo[m] * escala - base, alto[m] * escala - base
            ax.barh(k, d_bajo, color=COLOR_BAJO, height=0.6, label="Parámetro en su mínimo" if k == 0 else None)
            ax.barh(k, d_alto, color=COLOR_ALTO, height=0.6, label="Parámetro en su máximo" if k == 0 else None)
            # Etiqueta del valor del parámetro en el extremo de su barra (una a cada lado)
            izq, der = ((d_bajo, a), (d_alto, b)) if d_bajo <= d_alto else ((d_alto, b), (d_bajo, a))
            ax.annotate(f"{izq[1]:g}", (min(izq[0], 0), k), xytext=(-4, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8.5, color=TINTA_SEC)
            ax.annotate(f"{der[1]:g}", (max(der[0], 0), k), xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.5, color=TINTA_SEC)
        ax.axvline(0, color=TINTA, linewidth=1)
        ax.set_yticks(y, [ETIQUETAS_PARAMETROS[f[0]] for f in filas])
        ax.set_xlabel(f"Cambio respecto a la media base ({unidad})")
        ax.set_title(f"{titulo} (base = {base:,.1f})")
        ax.margins(x=0.18)
        ax.grid(axis="y", visible=False)
    ejes[0][0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(f"Análisis de sensibilidad un parámetro a la vez ({sens['n']} simulaciones por caso)",
                 fontweight="bold")
    return _guardar(fig, ruta)


def correlaciones(coef, ruta=None):
    """Barras agrupadas de la correlación de Spearman parámetro–métrica."""
    parametros = list(coef)
    if not parametros:
        return
    metricas = list(coef[parametros[0]])
    etiquetas = {"tasa_ataque": "Tasa de ataque", "pico_infectados": "Pico de I", "fallecidos": "Fallecidos"}
    ancho = 0.8 / len(metricas)
    x = np.arange(len(parametros))
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, m in enumerate(metricas):
        valores = [coef[p][m][0] for p in parametros]
        ax.bar(x + (k - (len(metricas) - 1) / 2) * ancho, valores, width=ancho * 0.92,
               color=COLORES_ESCENARIO[k], label=etiquetas.get(m, m))
    ax.axhline(0, color=TINTA, linewidth=1)
    ax.set_xticks(x, [ETIQUETAS_PARAMETROS[p] for p in parametros], rotation=12)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("ρ de Spearman")
    ax.set_title("Sensibilidad global: correlación entre parámetros muestreados y resultados")
    ax.legend(loc="upper right")
    return _guardar(fig, ruta)
