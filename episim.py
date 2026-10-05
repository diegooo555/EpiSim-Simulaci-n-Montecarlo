"""
EpiSim - Simulación Montecarlo de propagación de enfermedades contagiosas (Punto 4).

Comandos:
    python episim.py escenario config/con_vacunacion.csv          # Montecarlo de un escenario
    python episim.py comparar config/sin_vacunacion.csv config/con_vacunacion.csv
    python episim.py sensibilidad config/con_vacunacion.csv --n 200
    python episim.py traza config/con_vacunacion.csv              # Uso de cada R_i (1 simulación)
    python episim.py flujos                                       # Independencia entre simulaciones
    python episim.py equipo                                       # Cuadro técnico del equipo
    python episim.py todo                                         # Todo lo anterior
    python episim.py interfaz                                     # Interfaz gráfica

Opciones comunes: --n (simulaciones), --procesos, --salida (carpeta de resultados).
"""
import argparse
import csv
import os
import sys
import time

import analisis
import graficas
from aleatorio import (GENERADORES, FlujoAleatorio, crear_generador, max_flujos, saltar,
                       semilla_flujo, validar_independencia_flujos)
from configuracion import cargar_configuracion
from modelo import Bitacora, simular_epidemia
from montecarlo import ejecutar_montecarlo

CONFIG_SIN = os.path.join("config", "sin_vacunacion.csv")
CONFIG_CON = os.path.join("config", "con_vacunacion.csv")


def _encabezado(texto):
    print(f"\n{'═' * 78}\n  {texto}\n{'═' * 78}")


def _imprimir_configuracion(cfg):
    p = GENERADORES[cfg["generador"]]
    print(f"  Archivo: {cfg.origen}")
    print(f"  Generador: {p['nombre']} (a={p['a']}, c={p['c']}, m={p['m']})")
    for clave, valor in cfg.tabla():
        print(f"    {clave:<24}{valor}")


# ── Comandos ────────────────────────────────────────────────────────────

def cmd_escenario(ruta_cfg, n, procesos, salida):
    """Ejecuta el Montecarlo de un escenario, guarda tablas y genera sus gráficas."""
    cfg = cargar_configuracion(ruta_cfg)
    _encabezado(f"Escenario: {cfg['nombre']}")
    _imprimir_configuracion(cfg)
    n = n or cfg["n_simulaciones"]
    print(f"\n  Ejecutando {n} simulaciones de {cfg['dias']} días...")

    res = ejecutar_montecarlo(cfg, n, procesos)
    carpeta = os.path.join(salida, cfg["nombre"])
    res.guardar(carpeta)

    print("\n" + analisis.resumen_escenario(res))
    filas = analisis.estadisticas_descriptivas(res)
    print("\n  Estadísticas descriptivas")
    analisis.imprimir_tabla(filas)
    analisis.guardar_tabla(filas, os.path.join(carpeta, "estadisticas.csv"))

    coef = analisis.correlaciones_spearman(res)
    if coef:
        print("\n  Correlación de Spearman (parámetro vs. tasa de ataque)")
        for p, valores in coef.items():
            rho, pval = valores["tasa_ataque"]
            print(f"    {p:<24} ρ = {rho:+.3f}   (p = {pval:.2g})")

    print("\n  Generando gráficas...")
    graficas.curva_epidemica(res, os.path.join(carpeta, "curva_epidemica.png"))
    graficas.histogramas_escenario(res, carpeta)
    graficas.parametros_muestreados(res, os.path.join(carpeta, "parametros_muestreados.png"))
    graficas.validacion_pruebas(res, os.path.join(carpeta, "validacion_numeros.png"))
    graficas.correlaciones(coef, os.path.join(carpeta, "sensibilidad_spearman.png"))
    print(f"\n  Resultados guardados en: {carpeta}")
    return res


def cmd_comparar(rutas, n, procesos, salida):
    """Ejecuta varios escenarios y los compara (p. ej. sin vs. con vacunación)."""
    resultados = [cmd_escenario(r, n, procesos, salida) for r in rutas]
    carpeta = os.path.join(salida, "comparacion")
    os.makedirs(carpeta, exist_ok=True)
    _encabezado("Comparación de escenarios: " + " vs. ".join(r.nombre for r in resultados))

    filas = [f for r in resultados for f in analisis.estadisticas_descriptivas(r)]
    analisis.guardar_tabla(filas, os.path.join(carpeta, "estadisticas_por_escenario.csv"))
    for r in resultados:
        print(f"\n  {r.nombre}")
        analisis.imprimir_tabla(analisis.estadisticas_descriptivas(r))

    base = resultados[0]
    pruebas = []
    for otro in resultados[1:]:
        print(f"\n  {base.nombre} → {otro.nombre} (prueba de Mann-Whitney, bilateral)")
        for m in ("total_infectados", "pico_infectados", "fallecidos", "dia_pico"):
            d = analisis.prueba_diferencia(base, otro, m)
            if d:
                d["comparacion"] = f"{base.nombre} vs {otro.nombre}"
                pruebas.append(d)
                print(f"    {m:<18} media {d['media_a']:>9.1f} → {d['media_b']:>9.1f}   "
                      f"cambio {-d['reduccion_pct']:+6.1f} %   p = {d['p_valor']:.2g}")
    if pruebas:
        with open(os.path.join(carpeta, "pruebas_diferencia.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["comparacion", "metrica", "media_a", "media_b",
                                              "reduccion_pct", "U", "p_valor"])
            w.writeheader()
            w.writerows(pruebas)

    print("\n  Generando gráficas comparativas...")
    graficas.comparar_curvas(resultados, os.path.join(carpeta, "comparacion_curvas.png"))
    graficas.comparar_distribuciones(resultados, os.path.join(carpeta, "comparacion_distribuciones.png"))
    return resultados


def cmd_sensibilidad(ruta_cfg, n, procesos, salida):
    """Análisis de sensibilidad un parámetro a la vez (tornado)."""
    cfg = cargar_configuracion(ruta_cfg)
    n = n or 200
    _encabezado(f"Sensibilidad (un parámetro a la vez) - {cfg['nombre']}, {n} simulaciones por caso")
    t0 = time.perf_counter()
    sens = analisis.sensibilidad_oat(cfg, n, procesos)
    carpeta = os.path.join(salida, "sensibilidad")
    os.makedirs(carpeta, exist_ok=True)
    analisis.guardar_sensibilidad(sens, os.path.join(carpeta, f"sensibilidad_{cfg['nombre']}.csv"))

    print("\n  Ranking por impacto en la tasa de ataque (amplitud = |máximo − mínimo|)")
    orden = sorted(sens["parametros"], key=lambda f: abs(f[4]["tasa_ataque"] - f[3]["tasa_ataque"]), reverse=True)
    for k, (p, a, b, bajo, alto) in enumerate(orden, start=1):
        print(f"    {k}. {p:<22} {100 * abs(alto['tasa_ataque'] - bajo['tasa_ataque']):6.1f} puntos porcentuales")
    print(f"  Tiempo: {time.perf_counter() - t0:.1f} s")
    graficas.tornado(sens, os.path.join(carpeta, f"tornado_{cfg['nombre']}.png"))
    return sens


def cmd_traza(ruta_cfg, flujo, dias, salida):
    """Una simulación con la bitácora del uso de cada número pseudoaleatorio."""
    cfg = cargar_configuracion(ruta_cfg)
    semilla = semilla_flujo(cfg["generador"], cfg["semilla_maestra"], flujo)
    _encabezado(f"Traza de una simulación - {cfg['nombre']}, flujo {flujo}, semilla {semilla}")
    bitacora = Bitacora(max_lineas=600)
    rng = FlujoAleatorio(crear_generador(cfg["generador"], semilla), registrar=False)
    sim = simular_epidemia(cfg, rng, bitacora, dias_traza=dias)

    encabezado = [f"EpiSim - traza del uso de los números pseudoaleatorios",
                  f"Escenario: {cfg['nombre']} | Generador: {GENERADORES[cfg['generador']]['nombre']}",
                  f"Flujo {flujo}, semilla x0 = {semilla} | Se detallan la inicialización y los días 1 a {dias}",
                  ""]
    pie = ["", f"... ({bitacora.omitidas} líneas más omitidas)" if bitacora.omitidas else "",
           f"Resultado: total infectados = {sim['total_infectados']}, pico = {sim['pico_infectados']} "
           f"el día {sim['dia_pico']}, fallecidos = {sim['fallecidos']}, números usados = {rng.usados:,}"]
    texto = "\n".join(encabezado + bitacora.lineas + pie)
    print(texto)
    carpeta = os.path.join(salida, cfg["nombre"])
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"traza_flujo_{flujo}.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(f"\n  Traza guardada en: {ruta}")


def cmd_flujos(ruta_cfg, salida):
    """Verifica que los flujos de las distintas simulaciones no estén correlacionados."""
    cfg = cargar_configuracion(ruta_cfg)
    tipo, maestra = cfg["generador"], cfg["semilla_maestra"]
    n = max_flujos(tipo)
    p = GENERADORES[tipo]
    _encabezado(f"Independencia entre flujos - {p['nombre']}, {n} flujos, salto = {p['salto']:,}")
    aprobada, resultados, rho, primeros = validar_independencia_flujos(tipo, maestra, n, cfg["alpha"])
    print("  Secuencia: primer R_i de cada flujo (el que se usa para β)")
    for nombre, ok, est in resultados:
        print(f"    {nombre:<20}{est:>14.6f}   {'APROBADA' if ok else 'RECHAZADA'}")
    print(f"    Correlación entre flujos consecutivos: {rho:+.4f}")

    carpeta = os.path.join(salida, "flujos")
    os.makedirs(carpeta, exist_ok=True)
    if tipo == "lcg":
        # Contraste: con un salto potencia de 2 los flujos quedan correlacionados
        salto_malo = 2**21
        malos = [crear_generador(tipo, saltar(maestra, k * salto_malo, p["a"], p["c"], p["m"])).siguiente()
                 for k in range(n)]
        graficas.dispersion_flujos(malos, primeros, f"Salto 2^21 = {salto_malo:,} (descartado)",
                                   f"Salto 2^21 + 1 = {p['salto']:,} (usado)",
                                   os.path.join(carpeta, "dispersion_flujos.png"))
    return aprobada


def cmd_equipo(salida):
    _encabezado("Cuadro técnico del equipo")
    filas = analisis.especificaciones_equipo()
    for k, v in filas:
        print(f"  {k:<20}{v}")
    os.makedirs(salida, exist_ok=True)
    with open(os.path.join(salida, "equipo.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("caracteristica", "valor"), *filas])


# ── Principal ───────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="EpiSim - Montecarlo SEIR con pseudo_gen")
    sub = parser.add_subparsers(dest="comando", required=True)

    def comunes(p, con_n=True):
        if con_n:
            p.add_argument("--n", type=int, help="Simulaciones (por defecto: n_simulaciones del archivo)")
        p.add_argument("--procesos", type=int, help="Procesos en paralelo (por defecto: todos los núcleos)")
        p.add_argument("--salida", default="resultados", help="Carpeta de resultados")

    p = sub.add_parser("escenario", help="Montecarlo de un escenario")
    p.add_argument("config")
    comunes(p)
    p = sub.add_parser("comparar", help="Montecarlo de varios escenarios y comparación")
    p.add_argument("configs", nargs="+")
    comunes(p)
    p = sub.add_parser("sensibilidad", help="Sensibilidad un parámetro a la vez (tornado)")
    p.add_argument("config", nargs="?", default=CONFIG_CON)
    comunes(p)
    p = sub.add_parser("traza", help="Traza del uso de los R_i en una simulación")
    p.add_argument("config", nargs="?", default=CONFIG_CON)
    p.add_argument("--flujo", type=int, default=0, help="Número de flujo (simulación) a trazar")
    p.add_argument("--dias", type=int, default=2, help="Días con detalle")
    p.add_argument("--salida", default="resultados")
    p = sub.add_parser("flujos", help="Independencia entre los flujos de las simulaciones")
    p.add_argument("config", nargs="?", default=CONFIG_CON)
    p.add_argument("--salida", default="resultados")
    p = sub.add_parser("equipo", help="Cuadro técnico del equipo")
    p.add_argument("--salida", default="resultados")
    p = sub.add_parser("todo", help="Equipo, flujos, traza, comparación y sensibilidad")
    comunes(p)
    p.add_argument("--n-sensibilidad", type=int, default=200)
    sub.add_parser("interfaz", help="Abre la interfaz gráfica")

    args = parser.parse_args()
    if args.comando == "interfaz":
        # Tkinter y la interfaz solo se cargan cuando se solicita el modo gráfico.
        from gui import Aplicacion
        Aplicacion().mainloop()
        return 0

    try:
        if args.comando == "escenario":
            cmd_escenario(args.config, args.n, args.procesos, args.salida)
        elif args.comando == "comparar":
            cmd_comparar(args.configs, args.n, args.procesos, args.salida)
        elif args.comando == "sensibilidad":
            cmd_sensibilidad(args.config, args.n, args.procesos, args.salida)
        elif args.comando == "traza":
            cmd_traza(args.config, args.flujo, args.dias, args.salida)
        elif args.comando == "flujos":
            cmd_flujos(args.config, args.salida)
        elif args.comando == "equipo":
            cmd_equipo(args.salida)
        elif args.comando == "todo":
            t0 = time.perf_counter()
            cmd_equipo(args.salida)
            cmd_flujos(CONFIG_CON, args.salida)
            cmd_traza(CONFIG_CON, 0, 2, args.salida)
            cmd_comparar([CONFIG_SIN, CONFIG_CON], args.n, args.procesos, args.salida)
            cmd_sensibilidad(CONFIG_CON, args.n_sensibilidad, args.procesos, args.salida)
            print(f"\nTiempo total: {time.perf_counter() - t0:.1f} s")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
