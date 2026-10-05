"""
Modelo SEIR estocástico basado en individuos (EpiSim).

Estados:  S (susceptible) → E (expuesto) → I (infeccioso) → R (recuperado o fallecido)

Números pseudoaleatorios que consume una simulación, en orden:

  Inicialización
    1. Parámetros de la simulación (1 R_i c/u):  β, p, cobertura, efectividad, letalidad  ~ U(a, b)
    2. Casos índice: individuo = piso(R_i · N)  (se repite si sale uno ya infectado)
       + período infeccioso de cada caso índice
    3. Vacunación: para cada individuo i, R_i < cobertura → vacunado
       (factor de susceptibilidad = 1 − efectividad)

  Cada día t = 1..dias
    4. Progresión (paso determinístico, salvo la letalidad):
         E → I  si t = día programado de fin de incubación  → R_i para el período infeccioso
         I → R  si t = día programado de fin de infección    → R_i < letalidad → fallece
    5. Paso 1 del enunciado - contactos de cada infeccioso: R_i → n_contactos
    6. Paso 2 - selección del contactado: R_i → individuo piso(R_i · N)
    7. Paso 3 - transmisión: R_i < p · factor_vacunacion → el contactado pasa a E
         → R_i para su período de incubación

Los períodos se muestrean por individuo como enteros uniformes en [a, b] y, una vez
muestreados, la progresión es determinística (se programa en un calendario de eventos).
"""
import time

import numpy as np

from aleatorio import entero_uniforme, uniforme_ab
from configuracion import PARAMETROS_POR_SIMULACION

S, E, I, R = 0, 1, 2, 3
NOMBRES_ESTADOS = ("S", "E", "I", "R")


class Bitacora:
    """Registro legible (traza) de cómo se usa cada número pseudoaleatorio."""

    def __init__(self, max_lineas=400):
        self.lineas = []
        self.max_lineas = max_lineas
        self.omitidas = 0

    def agregar(self, texto):
        if len(self.lineas) < self.max_lineas:
            self.lineas.append(texto)
        else:
            self.omitidas += 1


class Poblacion:
    """
    Estado de los N individuos.

    Estructuras de datos:
        estado[i]          Estado del individuo i (S, E, I o R).
        susceptibilidad[i] Factor de vacunación: 1 si no está vacunado, 1 − efectividad si lo está.
        conteo             [S, E, I, R] actuales (se actualiza en cada transición).
        infecciosos        dict usado como conjunto ordenado de los individuos en I
                           (el orden de inserción hace la simulación reproducible).
        susceptibles       Lista de índices en S + posicion[i] para retirar en O(1)
                           (se usa si seleccion_contacto = "susceptibles").
        pasan_a_I/pasan_a_R  Calendario de eventos: día → lista de individuos.
    """

    def __init__(self, n):
        self.n = n
        self.estado = [S] * n
        self.susceptibilidad = [1.0] * n
        self.conteo = [n, 0, 0, 0]
        self.infecciosos = {}
        self.susceptibles = list(range(n))
        self.posicion = list(range(n))
        self.pasan_a_I = {}
        self.pasan_a_R = {}
        self.vacunados = 0
        self.total_infectados = 0
        self.infectados_vacunados = 0
        self.fallecidos = 0
        self.ids_fallecidos = []

    def _cambiar_estado(self, i, nuevo):
        self.conteo[self.estado[i]] -= 1
        self.conteo[nuevo] += 1
        self.estado[i] = nuevo

    def _retirar_susceptible(self, i):
        """Quita i de la lista de susceptibles intercambiándolo con el último (O(1))."""
        pos = self.posicion[i]
        ultimo = self.susceptibles[-1]
        self.susceptibles[pos] = ultimo
        self.posicion[ultimo] = pos
        self.susceptibles.pop()


def _programar(calendario, dia, i):
    calendario.setdefault(dia, []).append(i)


# ── Inicialización ──────────────────────────────────────────────────────

def muestrear_parametros(cfg, rng, bitacora=None):
    """Paso de inicialización 1: un R_i por parámetro, transformado a U(a, b)."""
    params = {}
    for nombre in PARAMETROS_POR_SIMULACION:
        a, b = cfg[nombre]
        r = rng()
        params[nombre] = uniforme_ab(r, a, b)
        if bitacora is not None:
            bitacora.agregar(f"[param] R={r:.6f} → {nombre} = {a} + ({b} − {a})·R = {params[nombre]:.4f}")
    return params


def exponer(pob, j, dia, cfg, rng, bitacora=None):
    """S → E: registra el contagio y programa el fin de la incubación."""
    a, b = cfg["periodo_incubacion"]
    r = rng()
    incubacion = entero_uniforme(r, a, b)
    pob._cambiar_estado(j, E)
    pob._retirar_susceptible(j)
    pob.total_infectados += 1
    if pob.susceptibilidad[j] < 1.0:
        pob.infectados_vacunados += 1
    _programar(pob.pasan_a_I, dia + incubacion, j)
    if bitacora is not None:
        bitacora.agregar(f"        R={r:.6f} → incubación = {a} + piso(R·{b - a + 1}) = {incubacion} días "
                         f"(pasa a I el día {dia + incubacion})")


def volver_infeccioso(pob, i, dia, cfg, rng, bitacora=None):
    """E → I (o caso índice): programa el fin del período infeccioso."""
    a, b = cfg["periodo_infeccioso"]
    r = rng()
    duracion = entero_uniforme(r, a, b)
    if pob.estado[i] == S:
        pob._retirar_susceptible(i)
    pob._cambiar_estado(i, I)
    pob.infecciosos[i] = None
    _programar(pob.pasan_a_R, dia + duracion, i)
    if bitacora is not None:
        bitacora.agregar(f"        R={r:.6f} → período infeccioso de {i} = {a} + piso(R·{b - a + 1}) = "
                         f"{duracion} días (pasa a R el día {dia + duracion})")


def inicializar_poblacion(cfg, params, rng, bitacora=None):
    """Pasos de inicialización 2 y 3: casos índice y vacunación."""
    n = cfg["poblacion"]
    pob = Poblacion(n)

    # Casos índice (no vacunados). Son infecciosos desde el día 1 (primer día con contactos).
    while pob.conteo[I] < cfg["infectados_iniciales"]:
        r = rng()
        i = int(r * n)
        if bitacora is not None:
            bitacora.agregar(f"[índice] R={r:.6f} → individuo piso(R·{n}) = {i}"
                             + ("" if pob.estado[i] == S else " (ya infectado, se repite)"))
        if pob.estado[i] == S:
            volver_infeccioso(pob, i, 1, cfg, rng, bitacora)
            pob.total_infectados += 1

    # Vacunación previa al brote: se consume un R_i por individuo aunque la cobertura sea 0,
    # para que los escenarios con y sin vacunación usen los mismos números en el resto
    # de la simulación (números aleatorios comunes).
    cobertura = params["cobertura_vacunacion"]
    factor = 1.0 - params["efectividad_vacuna"]
    for i in range(n):
        r = rng()
        if r < cobertura and pob.estado[i] == S:
            pob.susceptibilidad[i] = factor
            pob.vacunados += 1
            if bitacora is not None and pob.vacunados <= 5:
                bitacora.agregar(f"[vacuna] R={r:.6f} < cobertura {cobertura:.4f} → individuo {i} vacunado "
                                 f"(factor de susceptibilidad {factor:.4f})")
    if bitacora is not None:
        bitacora.agregar(f"[vacuna] ... {n} R_i usados; {pob.vacunados} vacunados ({pob.vacunados / n:.1%})")
    return pob


# ── Mecánica diaria ─────────────────────────────────────────────────────

def generar_contactos(beta, rng, modo="estocastico", bitacora=None):
    """
    Paso 1: número de contactos de un infeccioso en el día.

    modo "literal":     piso(r · β)      → con β < 1 siempre da 0 (la epidemia no avanza).
    modo "estocastico": piso(r + β)      → vale piso(β) o piso(β)+1, con media exactamente β.
                                            Con β ∈ [0.3, 0.5]: 1 contacto con probabilidad β.
    """
    r = rng()
    n = int(r * beta) if modo == "literal" else int(r + beta)
    if bitacora is not None:
        formula = f"piso(R·{beta:.4f})" if modo == "literal" else f"piso(R + {beta:.4f})"
        bitacora.agregar(f"    R={r:.6f} → contactos = {formula} = {n}")
    return n


def seleccionar_contacto(pob, rng, modo="poblacion", bitacora=None):
    """
    Paso 2: elige al contactado.

    modo "poblacion":    individuo piso(r·N) de toda la población; si no es susceptible,
                         el contacto no produce contagio (mezcla homogénea, acción de masas).
    modo "susceptibles": individuo al azar entre los susceptibles actuales (lectura literal
                         del enunciado; la fuerza de infección no disminuye al agotarse S).

    Returns:
        int | None: índice del susceptible contactado, o None si el contacto no es con un S.
    """
    r = rng()
    if modo == "susceptibles":
        if not pob.susceptibles:
            return None
        j = pob.susceptibles[int(r * len(pob.susceptibles))]
    else:
        j = int(r * pob.n)
    es_s = pob.estado[j] == S
    if bitacora is not None:
        bitacora.agregar(f"    R={r:.6f} → contacto con individuo {j} "
                         f"({NOMBRES_ESTADOS[pob.estado[j]]}{', vacunado' if pob.susceptibilidad[j] < 1 else ''})"
                         + ("" if es_s else " → no es susceptible, sin contagio"))
    return j if es_s else None


def evaluar_transmision(pob, j, p_base, rng, bitacora=None):
    """Paso 3: contagio si r < p_base · factor_vacunacion(j)."""
    r = rng()
    p_efectiva = p_base * pob.susceptibilidad[j]
    contagio = r < p_efectiva
    if bitacora is not None:
        bitacora.agregar(f"    R={r:.6f} {'<' if contagio else '≥'} p_efectiva = {p_base:.4f}·"
                         f"{pob.susceptibilidad[j]:.4f} = {p_efectiva:.4f} → "
                         + ("CONTAGIO (S → E)" if contagio else "sin contagio"))
    return contagio


def progresar_estados(pob, dia, cfg, params, rng, bitacora=None):
    """Transiciones programadas para el día: I → R (con letalidad) y E → I."""
    for i in pob.pasan_a_R.pop(dia, ()):
        pob._cambiar_estado(i, R)
        del pob.infecciosos[i]
        r = rng()
        fallece = r < params["letalidad"]
        if fallece:
            pob.fallecidos += 1
            pob.ids_fallecidos.append(i)
        if bitacora is not None:
            bitacora.agregar(f"  Individuo {i}: I → R | R={r:.6f} "
                             f"{'<' if fallece else '≥'} letalidad {params['letalidad']:.4f} → "
                             + ("fallece" if fallece else "se recupera"))

    for i in pob.pasan_a_I.pop(dia, ()):
        if bitacora is not None:
            bitacora.agregar(f"  Individuo {i}: E → I (fin de incubación)")
        volver_infeccioso(pob, i, dia, cfg, rng, bitacora)


def propagar_contagio(pob, dia, cfg, params, rng, bitacora=None):
    """Pasos 1-3 para cada individuo infeccioso del día."""
    beta = params["beta"]
    p_base = params["p_transmision"]
    modo_contactos = cfg["modo_contactos"]
    modo_seleccion = cfg["seleccion_contacto"]

    # Se itera sobre una copia: quienes se contagian hoy entran a E, no a I.
    for i in list(pob.infecciosos):
        if bitacora is not None:
            bitacora.agregar(f"  Infeccioso {i}:")
        for _ in range(generar_contactos(beta, rng, modo_contactos, bitacora)):
            j = seleccionar_contacto(pob, rng, modo_seleccion, bitacora)
            if j is not None and evaluar_transmision(pob, j, p_base, rng, bitacora):
                exponer(pob, j, dia, cfg, rng, bitacora)


# ── Simulación completa ─────────────────────────────────────────────────

class SimulacionPasoAPaso:
    """
    Simulación que avanza un día a la vez (la usan simular_epidemia y la interfaz gráfica).

    Atributos útiles:
        params  Parámetros muestreados.
        pob     Población (estado de cada individuo).
        dia     Último día simulado (0 = estado inicial).
        serie   Lista de conteos (S, E, I, R) de los días 0..dia.
    """

    def __init__(self, cfg, rng, bitacora=None):
        self.cfg = cfg
        self.rng = rng
        self.params = muestrear_parametros(cfg, rng, bitacora)
        self.pob = inicializar_poblacion(cfg, params=self.params, rng=rng, bitacora=bitacora)
        self.dia = 0
        self.serie = [tuple(self.pob.conteo)]   # Día 0

    @property
    def terminada(self):
        return self.dia >= self.cfg["dias"]

    @property
    def extinguida(self):
        """Sin expuestos ni infecciosos: el estado ya no puede cambiar."""
        return self.pob.conteo[E] + self.pob.conteo[I] == 0

    def avanzar(self, bitacora=None):
        """Simula el día siguiente: progresión de estados y luego contagios."""
        if self.terminada:
            return
        self.dia += 1
        if bitacora is not None:
            bitacora.agregar(f"\n══ Día {self.dia} ══")
        progresar_estados(self.pob, self.dia, self.cfg, self.params, self.rng, bitacora)
        propagar_contagio(self.pob, self.dia, self.cfg, self.params, self.rng, bitacora)
        self.serie.append(tuple(self.pob.conteo))
        if bitacora is not None:
            bitacora.agregar(f"  Fin del día {self.dia}: " + ", ".join(
                f"{nombre}={valor}" for nombre, valor in zip(NOMBRES_ESTADOS, self.pob.conteo)))

    def completar(self):
        """Rellena los días restantes cuando el brote ya se extinguió (no consume números)."""
        faltan = self.cfg["dias"] - self.dia
        self.serie.extend([tuple(self.pob.conteo)] * faltan)
        self.dia += faltan


def simular_epidemia(cfg, rng, bitacora=None, dias_traza=3):
    """
    Ejecuta una simulación completa.

    Args:
        cfg (Configuracion): Parámetros del escenario.
        rng (callable): Fuente de R_i ~ U(0,1) (FlujoAleatorio).
        bitacora (Bitacora, opcional): Si se pasa, registra el uso de cada R_i
            en la inicialización y en los primeros `dias_traza` días.

    Returns:
        dict: serie (array (dias+1) × 4 con S, E, I, R), parámetros muestreados y métricas.
    """
    t0 = time.perf_counter()
    sim = SimulacionPasoAPaso(cfg, rng, bitacora)
    while not sim.terminada:
        if sim.extinguida:
            sim.completar()     # Brote extinguido: el estado ya no cambia hasta el final
            break
        sim.avanzar(bitacora if (bitacora is not None and sim.dia < dias_traza) else None)

    serie = np.array(sim.serie, dtype=np.int32)
    resultado = {"serie": serie, "parametros": sim.params}
    resultado.update(calcular_metricas(serie, sim.pob, cfg))
    resultado["tiempo_s"] = time.perf_counter() - t0
    return resultado


def calcular_metricas(serie, pob, cfg):
    """Métricas de resumen de una simulación."""
    infecciosos = serie[:, I]
    dia_pico = int(np.argmax(infecciosos))
    activos = serie[:, E] + serie[:, I]
    controlados = np.nonzero(activos[dia_pico:] <= cfg["umbral_control"])[0]
    return {
        "pico_infectados": int(infecciosos[dia_pico]),
        "dia_pico": dia_pico,
        "dia_control": float(dia_pico + controlados[0]) if len(controlados) else float("nan"),
        "total_infectados": pob.total_infectados,
        "tasa_ataque": pob.total_infectados / pob.n,
        "fallecidos": pob.fallecidos,
        "vacunados": pob.vacunados,
        "infectados_vacunados": pob.infectados_vacunados,
        "susceptibles_finales": int(serie[-1, S]),
    }
