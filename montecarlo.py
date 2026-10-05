"""
Ejecución Montecarlo de un escenario: muchas simulaciones independientes de EpiSim.

Flujo por simulación (flujo k = 0, 1, 2, ...):
    semilla_k = semilla maestra avanzada k·SALTO pasos (o la k-ésima del archivo de semillas)
      → generador de pseudo_gen → simulación SEIR → R_i usados → 5 pruebas estadísticas
      → si aprueba: se ACEPTA;  si falla alguna prueba: se RECHAZA y se toma el flujo siguiente.

Se toman flujos en orden hasta completar n_simulaciones aceptadas. Las simulaciones corren en
paralelo (multiprocessing), pero los resultados se procesan en el orden de k, así que el
resultado es idéntico con 1 o con varios procesos.
"""
import csv
import math
import os
import time
from multiprocessing import Pool

import numpy as np

from aleatorio import (GENERADORES, FlujoAleatorio, PRUEBAS, crear_generador, max_flujos,
                       semilla_flujo, validar_secuencia)
from configuracion import PARAMETROS_POR_SIMULACION
from modelo import simular_epidemia
from pseudo_gen.utilidades.archivo_manager import ArchivoManager

METRICAS = ["total_infectados", "tasa_ataque", "fallecidos", "pico_infectados", "dia_pico",
            "dia_control", "vacunados", "infectados_vacunados", "susceptibles_finales"]


class ResultadosEscenario:
    """Resultados de las simulaciones aceptadas y rechazadas de un escenario."""

    def __init__(self, cfg, aceptadas, rechazadas, tiempo_total, procesos):
        self.cfg = cfg
        self.nombre = cfg["nombre"]
        self.aceptadas = aceptadas
        self.rechazadas = rechazadas
        self.tiempo_total = tiempo_total
        self.procesos = procesos
        # series[k, t, estado]: n_sim × (dias + 1) × 4
        self.series = np.stack([s["serie"] for s in aceptadas]) if aceptadas else np.empty((0, 0, 4))

    def columna(self, clave):
        """Arreglo con una métrica o parámetro muestreado de todas las simulaciones aceptadas."""
        if clave in PARAMETROS_POR_SIMULACION:
            return np.array([s["parametros"][clave] for s in self.aceptadas], dtype=float)
        return np.array([s[clave] for s in self.aceptadas], dtype=float)

    def guardar(self, carpeta):
        """Guarda simulaciones.csv, rechazadas.csv, series.npz y la configuración usada."""
        os.makedirs(carpeta, exist_ok=True)
        self.cfg.guardar(os.path.join(carpeta, "configuracion_usada.csv"))
        np.savez_compressed(os.path.join(carpeta, "series.npz"), series=self.series,
                            semillas=np.array([s["semilla"] for s in self.aceptadas], dtype=np.int64))

        with open(os.path.join(carpeta, "simulaciones.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["flujo", "semilla", *PARAMETROS_POR_SIMULACION, *METRICAS,
                        "numeros_usados", "tiempo_simulacion_s", "tiempo_validacion_s"])
            for s in self.aceptadas:
                w.writerow([s["flujo"], s["semilla"],
                            *(f"{s['parametros'][p]:.6f}" for p in PARAMETROS_POR_SIMULACION),
                            *(_formato(s[m]) for m in METRICAS),
                            s["numeros_usados"], f"{s['tiempo_s']:.4f}", f"{s['tiempo_validacion_s']:.4f}"])

        with open(os.path.join(carpeta, "rechazadas.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["flujo", "semilla", "numeros_usados", "motivo",
                        *(f"estadistico_{n}" for n, _ in PRUEBAS)])
            for s in self.rechazadas:
                w.writerow([s["flujo"], s["semilla"], s["numeros_usados"], s["motivo"],
                            *(f"{est:.6f}" for _, _, est in s.get("validacion", []))])


def _formato(valor):
    if isinstance(valor, float):
        return "" if math.isnan(valor) else f"{valor:.6f}"
    return valor


def _simular_flujo(tarea):
    """Trabajo de un proceso: una simulación completa + validación de sus números."""
    cfg, flujo, semilla = tarea
    validar = cfg["validar_numeros"] == "si"
    try:
        rng = FlujoAleatorio(crear_generador(cfg["generador"], semilla), registrar=validar)
    except ValueError as e:                      # Semilla fuera del rango del generador
        return {"flujo": flujo, "semilla": semilla, "aceptada": False, "numeros_usados": 0,
                "motivo": f"semilla inválida ({e})"}

    sim = simular_epidemia(cfg, rng)
    sim.update(flujo=flujo, semilla=semilla, numeros_usados=rng.usados, tiempo_validacion_s=0.0,
               aceptada=True, motivo="")

    if rng.usados > GENERADORES[cfg["generador"]]["salto"] and not cfg["archivo_semillas"]:
        sim["motivo"] = "ADVERTENCIA: usó más números que el tamaño del flujo (solapamiento)"

    if validar:
        t0 = time.perf_counter()
        aprobada, resultados = validar_secuencia(rng.registro, cfg["alpha"])
        sim["tiempo_validacion_s"] = time.perf_counter() - t0
        sim["validacion"] = resultados
        if not aprobada:
            sim["aceptada"] = False
            sim["motivo"] = "falló: " + ", ".join(n for n, ok, _ in resultados if not ok)
            del sim["serie"]                     # No se usa; se libera memoria
    return sim


def _tareas(cfg):
    """Genera (cfg, flujo, semilla) para todos los flujos disponibles, en orden."""
    if cfg["archivo_semillas"]:
        semillas = ArchivoManager.leer_semillas(cfg["archivo_semillas"])
        for k, s in enumerate(semillas):
            yield cfg, k, s
    else:
        tipo = cfg["generador"]
        for k in range(max_flujos(tipo)):
            yield cfg, k, semilla_flujo(tipo, cfg["semilla_maestra"], k)


def ejecutar_montecarlo(cfg, n_simulaciones=None, procesos=None, mostrar_progreso=True, progreso=None):
    """
    Ejecuta simulaciones hasta tener n_simulaciones aceptadas.

    Args:
        cfg (Configuracion): Escenario.
        n_simulaciones (int, opcional): Por defecto cfg["n_simulaciones"].
        procesos (int, opcional): Procesos en paralelo (por defecto, todos los núcleos).
        mostrar_progreso (bool): Imprime el avance cada 10 %.
        progreso (callable, opcional): progreso(aceptadas, rechazadas, n_objetivo) tras cada simulación
            (lo usa la interfaz gráfica).

    Returns:
        ResultadosEscenario
    """
    n_objetivo = n_simulaciones or cfg["n_simulaciones"]
    procesos = procesos or os.cpu_count() or 1
    aceptadas, rechazadas = [], []
    siguiente_aviso = 0.1
    t0 = time.perf_counter()

    def procesar(sim):
        nonlocal siguiente_aviso
        (aceptadas if sim["aceptada"] else rechazadas).append(sim)
        avance = len(aceptadas) / n_objetivo
        if progreso is not None:
            progreso(len(aceptadas), len(rechazadas), n_objetivo)
        if mostrar_progreso and avance >= siguiente_aviso:
            print(f"  [{cfg['nombre']}] {len(aceptadas)}/{n_objetivo} aceptadas, "
                  f"{len(rechazadas)} rechazadas ({time.perf_counter() - t0:.1f} s)")
            siguiente_aviso += 0.1
        return len(aceptadas) >= n_objetivo

    if procesos == 1:
        for tarea in _tareas(cfg):
            if procesar(_simular_flujo(tarea)):
                break
    else:
        with Pool(procesos) as pool:     # Al salir del bloque se terminan los procesos pendientes
            for sim in pool.imap(_simular_flujo, _tareas(cfg), chunksize=2):
                if procesar(sim):
                    break

    if len(aceptadas) < n_objetivo:
        print(f"  ADVERTENCIA: solo {len(aceptadas)} simulaciones aceptadas de {n_objetivo} solicitadas "
              f"(se agotaron los flujos/semillas disponibles).")
    return ResultadosEscenario(cfg, aceptadas, rechazadas, time.perf_counter() - t0, procesos)
