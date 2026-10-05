"""
Análisis de resultados de EpiSim: estadísticas descriptivas, comparación de escenarios,
análisis de sensibilidad y especificaciones del equipo.
"""
import csv
import ctypes
import math
import os
import platform
import sys

import numpy as np
from scipy import stats

from montecarlo import ejecutar_montecarlo

# Métricas que se resumen en la tabla descriptiva: (clave, etiqueta)
METRICAS_TABLA = [
    ("total_infectados", "Total infectados"),
    ("tasa_ataque", "Tasa de ataque"),
    ("fallecidos", "Fallecidos"),
    ("pico_infectados", "Pico de I (personas)"),
    ("dia_pico", "Día del pico"),
    ("dia_control", "Día de control"),
    ("infectados_vacunados", "Infectados vacunados"),
    ("tiempo_s", "Tiempo por simulación (s)"),
]

# Parámetros que se varían en el análisis de sensibilidad (uno a la vez)
PARAMETROS_SENSIBILIDAD = ["beta", "p_transmision", "cobertura_vacunacion", "efectividad_vacuna",
                           "periodo_infeccioso", "periodo_incubacion"]

ETIQUETAS_PARAMETROS = {
    "beta": "β (contactos/día)",
    "p_transmision": "p (transmisión)",
    "cobertura_vacunacion": "Cobertura de vacunación",
    "efectividad_vacuna": "Efectividad de la vacuna",
    "periodo_infeccioso": "Período infeccioso (días)",
    "periodo_incubacion": "Período de incubación (días)",
    "letalidad": "Letalidad",
}


# ── Estadísticas descriptivas ───────────────────────────────────────────

def estadisticas_descriptivas(res):
    """
    Resumen de cada métrica: n, media, desviación, IC 95 % de la media, mínimo, percentiles, máximo.

    Para "día de control" solo se usan las simulaciones en las que el brote se controló
    dentro del horizonte; la columna n_validos indica cuántas fueron.
    """
    filas = []
    for clave, etiqueta in METRICAS_TABLA:
        x = res.columna(clave)
        validos = x[~np.isnan(x)]
        n = len(validos)
        if n == 0:
            filas.append({"escenario": res.nombre, "metrica": etiqueta, "n_validos": 0})
            continue
        media = validos.mean()
        de = validos.std(ddof=1) if n > 1 else 0.0
        mitad_ic = stats.t.ppf(0.975, n - 1) * de / math.sqrt(n) if n > 1 else 0.0
        filas.append({
            "escenario": res.nombre,
            "metrica": etiqueta,
            "n_validos": n,
            "media": media,
            "desviacion": de,
            "ic95_inf": media - mitad_ic,
            "ic95_sup": media + mitad_ic,
            "minimo": validos.min(),
            "p2_5": np.percentile(validos, 2.5),
            "mediana": np.median(validos),
            "p97_5": np.percentile(validos, 97.5),
            "maximo": validos.max(),
        })
    return filas


def imprimir_tabla(filas):
    print(f"  {'Métrica':<27}{'n':>6}{'Media':>11}{'Desv.':>10}{'IC95% media':>22}"
          f"{'P2.5':>10}{'Mediana':>10}{'P97.5':>10}")
    for f in filas:
        if not f["n_validos"]:
            print(f"  {f['metrica']:<27}{0:>6}   (sin datos)")
            continue
        dec = 4 if f["media"] < 1 else 1
        ic = f"[{f['ic95_inf']:.{dec}f}, {f['ic95_sup']:.{dec}f}]"
        print(f"  {f['metrica']:<27}{f['n_validos']:>6}{f['media']:>11.{dec}f}{f['desviacion']:>10.{dec}f}"
              f"{ic:>22}{f['p2_5']:>10.{dec}f}{f['mediana']:>10.{dec}f}{f['p97_5']:>10.{dec}f}")


def guardar_tabla(filas, ruta):
    columnas = list(dict.fromkeys(k for f in filas for k in f))
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        for f in filas:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, (float, np.floating)) else v) for k, v in f.items()})


def resumen_escenario(res):
    """Texto con los datos generales de la ejecución."""
    n = len(res.aceptadas)
    total = n + len(res.rechazadas)
    no_controladas = int(np.isnan(res.columna("dia_control")).sum()) if n else 0
    numeros = res.columna("numeros_usados") if n else np.array([0])
    return (f"  Simulaciones aceptadas: {n} de {total} ejecutadas "
            f"({len(res.rechazadas)} rechazadas por las pruebas de los R_i)\n"
            f"  Brotes no controlados dentro de {res.cfg['dias']} días: {no_controladas}\n"
            f"  Números pseudoaleatorios por simulación: media {numeros.mean():,.0f}, máx. {numeros.max():,.0f}\n"
            f"  Tiempo total: {res.tiempo_total:.1f} s con {res.procesos} proceso(s)")


def prueba_diferencia(res_a, res_b, clave):
    """
    Compara una métrica entre dos escenarios con la prueba de Mann-Whitney (no paramétrica:
    las distribuciones son asimétricas y a veces bimodales).
    """
    a = res_a.columna(clave)
    b = res_b.columna(clave)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"metrica": clave, "media_a": a.mean(), "media_b": b.mean(),
            "reduccion_pct": 100 * (1 - b.mean() / a.mean()) if a.mean() else float("nan"),
            "U": u, "p_valor": p}


# ── Sensibilidad ────────────────────────────────────────────────────────

def correlaciones_spearman(res, metricas=("tasa_ataque", "pico_infectados", "fallecidos")):
    """
    Sensibilidad global: correlación de Spearman entre cada parámetro muestreado por simulación
    y cada métrica, usando las simulaciones Montecarlo del escenario (sin corridas adicionales).
    """
    resultado = {}
    for p in ["beta", "p_transmision", "cobertura_vacunacion", "efectividad_vacuna", "letalidad"]:
        x = res.columna(p)
        if np.ptp(x) == 0:          # Parámetro fijo en el escenario
            continue
        resultado[p] = {}
        for m in metricas:
            y = res.columna(m)
            rho, pval = stats.spearmanr(x, y)
            resultado[p][m] = (rho, pval)
    return resultado


def sensibilidad_oat(cfg, n_simulaciones, procesos=None,
                     metricas=("tasa_ataque", "pico_infectados", "fallecidos"), progreso=None):
    """
    Sensibilidad local, un parámetro a la vez (gráfico tornado).

    Para cada parámetro se fija su valor en el extremo inferior y luego en el superior de su
    rango (los demás siguen muestreándose de sus distribuciones) y se ejecutan n_simulaciones.
    Todas las corridas usan los mismos flujos de números (números aleatorios comunes), de modo
    que la diferencia se debe al parámetro y no al azar.

    progreso (callable, opcional): progreso(caso_actual, total_casos, descripcion).

    Returns:
        dict: {"base": {metrica: media}, "parametros": [(nombre, a, b, {metrica: media_a},
               {metrica: media_b}), ...], "metricas": metricas, "n": n_simulaciones}
    """
    def medias(res):
        return {m: float(np.nanmean(res.columna(m))) for m in metricas}

    variables = [p for p in PARAMETROS_SENSIBILIDAD if cfg[p][0] != cfg[p][1]]
    total_casos = 1 + 2 * len(variables)
    caso = 0

    def avisar(descripcion):
        nonlocal caso
        if progreso is not None:
            progreso(caso, total_casos, descripcion)
        caso += 1

    avisar("caso base")

    print(f"\n  Caso base ({n_simulaciones} simulaciones)")
    base = medias(ejecutar_montecarlo(cfg.copia(nombre=f"{cfg['nombre']}_base"), n_simulaciones,
                                      procesos, mostrar_progreso=False))
    print("    " + ", ".join(f"{m} = {v:.4f}" for m, v in base.items()))

    filas = []
    for p in PARAMETROS_SENSIBILIDAD:
        a, b = cfg[p]
        if a == b:
            print(f"  {p}: fijo en el escenario ({a}), se omite")
            continue
        extremos = []
        for valor in (a, b):
            avisar(f"{p} = {valor:g}")
            r = ejecutar_montecarlo(cfg.copia(**{p: (valor, valor)}, nombre=f"{cfg['nombre']}_{p}_{valor}"),
                                    n_simulaciones, procesos, mostrar_progreso=False)
            extremos.append(medias(r))
        filas.append((p, a, b, extremos[0], extremos[1]))
        m = metricas[0]
        print(f"  {p:<22} {m}: {a:>6} → {extremos[0][m]:.4f}   |   {b:>6} → {extremos[1][m]:.4f}")

    if progreso is not None:
        progreso(total_casos, total_casos, "terminado")
    return {"base": base, "metricas": list(metricas), "n": n_simulaciones, "parametros": filas}


def guardar_sensibilidad(sens, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["parametro", "valor_bajo", "valor_alto", "metrica", "media_bajo", "media_alto",
                    "amplitud", "media_base"])
        for p, a, b, bajo, alto in sens["parametros"]:
            for m in sens["metricas"]:
                w.writerow([p, a, b, m, f"{bajo[m]:.6f}", f"{alto[m]:.6f}",
                            f"{abs(alto[m] - bajo[m]):.6f}", f"{sens['base'][m]:.6f}"])


# ── Especificaciones del equipo ─────────────────────────────────────────

def _memoria_gb():
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1024**3
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return float("nan")


def _procesador():
    if sys.platform == "win32":
        try:
            import winreg
            clave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(clave, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def especificaciones_equipo():
    """Cuadro técnico: lenguaje, sistema operativo, procesador y memoria."""
    import matplotlib
    import scipy
    return [
        ("Lenguaje", f"Python {platform.python_version()} ({platform.python_implementation()})"),
        ("Librerías", f"NumPy {np.__version__}, SciPy {scipy.__version__}, Matplotlib {matplotlib.__version__}"),
        ("Sistema operativo", f"{platform.system()} {platform.release()} ({platform.version()})"),
        ("Procesador", _procesador()),
        ("Núcleos lógicos", str(os.cpu_count())),
        ("Memoria RAM", f"{_memoria_gb():.1f} GB"),
    ]
