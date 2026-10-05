"""
Integración del módulo pseudo_gen con EpiSim.

Todos los números aleatorios del simulador salen de un generador congruencial de pseudo_gen.
No se usa random ni numpy.random.

Flujos independientes (sin solapamiento)
----------------------------------------
Cada simulación Montecarlo usa su propio flujo de números. Si se tomaran semillas "al azar",
dos simulaciones podrían reutilizar el mismo tramo del ciclo del generador. Para evitarlo,
la semilla de la simulación k es el estado del generador después de k·SALTO pasos desde la
semilla maestra (salto adelante, "jump-ahead"):

    x_{n+k} = A_k · x_n + C_k  (mod m),   A_k = a^k,   C_k = c·(a^k − 1)/(a − 1)

A_k y C_k se calculan por exponenciación binaria en O(log k). Así el flujo k ocupa exactamente
los números [k·SALTO, (k+1)·SALTO) del ciclo y ningún par de simulaciones comparte números
mientras cada una use menos de SALTO números (una simulación típica usa ~2·10^5).

¿Por qué SALTO = 2^21 + 1 y no 2^21 en el LCG?
Las semillas de los flujos forman a su vez un congruencial: s_{k+1} = A·s_k + C con A = a^SALTO.
Con m = 2^32 y SALTO = 2^21 se cumple A ≡ 1 (mod 2^23), así que el número j del flujo k+1 es
casi el del flujo k más una constante: las simulaciones quedarían correlacionadas (se verificó:
la secuencia "primer R_i de cada flujo" falla las 5 pruebas en 20 de 20 bloques). Con un salto
impar A deja de estar cerca de 1 y esa secuencia aprueba en la proporción esperada por azar
(ver validar_independencia_flujos).

Validación
----------
FlujoAleatorio registra cada R_i entregado. Al terminar la simulación, esa secuencia exacta
(los números que realmente se usaron) se somete a las 5 pruebas de pseudo_gen.
"""
import pseudo_gen as pg

# Parámetros de los generadores (ver pseudo_gen/README.md, "Parámetros recomendados").
GENERADORES = {
    "lcg": {
        "nombre": "Congruencial Lineal (Numerical Recipes)",
        "a": 1664525, "c": 1013904223, "m": 2**32,
        "salto": 2**21 + 1,   # 2047 flujos de 2.097.153 números
    },
    "mult": {
        "nombre": "Congruencial Multiplicativo (MINSTD, Park & Miller)",
        "a": 16807, "c": 0, "m": 2**31 - 1,
        "salto": 2**20,   # 2047 flujos de 1.048.576 números
    },
}

# Las 5 pruebas del módulo, en el orden en que se ejecutan
PRUEBAS = [
    ("Medias", pg.PruebaMedias),
    ("Varianza", pg.PruebaVarianza),
    ("Chi-cuadrado", pg.PruebaChi2),
    ("Kolmogorov-Smirnov", pg.PruebaKS),
    ("Póker", pg.PruebaPoker),
]


# ── Generadores y semillas ──────────────────────────────────────────────

def crear_generador(tipo, semilla):
    """Instancia el generador de pseudo_gen indicado con la semilla dada."""
    p = GENERADORES[tipo]
    if p["c"] == 0:
        return pg.CongruencialMultiplicativo(x0=semilla, a=p["a"], m=p["m"])
    return pg.CongruencialLineal(x0=semilla, a=p["a"], c=p["c"], m=p["m"])


def saltar(x, k, a, c, m):
    """
    Estado del congruencial x_{n+k} a partir de x_n sin generar los k números intermedios.

    Compone la transformación f(x) = a·x + c consigo misma por exponenciación binaria:
    si f^j(x) = A·x + C, entonces f^{2j}(x) = A²·x + C·(A + 1).
    """
    A_total, C_total = 1, 0     # Transformación acumulada (identidad)
    A_pot, C_pot = a % m, c % m  # f^(2^i)
    while k > 0:
        if k & 1:
            A_total = (A_pot * A_total) % m
            C_total = (A_pot * C_total + C_pot) % m
        C_pot = (C_pot * (A_pot + 1)) % m
        A_pot = (A_pot * A_pot) % m
        k >>= 1
    return (A_total * x + C_total) % m


def semilla_flujo(tipo, semilla_maestra, indice):
    """Semilla (estado inicial) del flujo `indice`: la maestra avanzada indice·SALTO pasos."""
    p = GENERADORES[tipo]
    return saltar(semilla_maestra, indice * p["salto"], p["a"], p["c"], p["m"])


def max_flujos(tipo):
    """Número de flujos disjuntos que caben en el período del generador."""
    p = GENERADORES[tipo]
    return (p["m"] - (1 if p["c"] == 0 else 0)) // p["salto"]


def validar_independencia_flujos(tipo, semilla_maestra, n_flujos, alpha=0.05):
    """
    Verifica que los flujos no estén correlacionados entre sí.

    Toma el primer R_i de cada uno de los n_flujos flujos (el que se usa para β) y somete esa
    secuencia a las 5 pruebas, además de calcular la correlación entre flujos consecutivos.

    Returns:
        tuple: (aprobada, resultados_pruebas, correlacion_lag1, primeros_ri)
    """
    primeros = []
    for k in range(n_flujos):
        primeros.append(crear_generador(tipo, semilla_flujo(tipo, semilla_maestra, k)).siguiente())
    aprobada, resultados = validar_secuencia(primeros, alpha)
    n = len(primeros)
    media = sum(primeros) / n
    cov = sum((primeros[i] - media) * (primeros[i + 1] - media) for i in range(n - 1)) / (n - 1)
    var = sum((r - media) ** 2 for r in primeros) / n
    return aprobada, resultados, cov / var, primeros


# ── Flujo de números para una simulación ────────────────────────────────

class FlujoAleatorio:
    """
    Fuente de R_i ~ U(0, 1) para una simulación. Se usa como función: r = flujo().

    Args:
        generador: Generador de pseudo_gen (con método siguiente()).
        registrar (bool): Si True guarda cada R_i entregado para validarlos al final.
    """

    def __init__(self, generador, registrar=True):
        self.generador = generador
        self.registro = [] if registrar else None
        self._siguiente = generador.siguiente
        self._contador = 0

    def __call__(self):
        r = self._siguiente()
        self._contador += 1
        if self.registro is not None:
            self.registro.append(r)
        return r

    @property
    def usados(self):
        """Cantidad de números consumidos hasta ahora."""
        return self._contador


# ── Transformaciones ────────────────────────────────────────────────────

def uniforme_ab(r, a, b):
    """U(a, b) a partir de r ~ U(0,1), con la transformación de pseudo_gen: a + (b − a)·r."""
    if a == b:
        return a
    return pg.uniforme([r], a, b)[0]


def entero_uniforme(r, a, b):
    """Uniforme discreta en {a, a+1, ..., b} a partir de r ~ U(0,1): a + piso(r·(b − a + 1))."""
    return a + int(r * (b - a + 1))


# ── Validación ──────────────────────────────────────────────────────────

def validar_secuencia(ri, alpha=0.05):
    """
    Aplica las 5 pruebas de pseudo_gen a la secuencia ri.

    Returns:
        tuple: (aprobada, [(nombre_prueba, aprobada, estadistico), ...])
    """
    resultados = []
    for nombre, Prueba in PRUEBAS:
        res = Prueba(ri, alpha=alpha).ejecutar()
        resultados.append((nombre, bool(res.aprobada), float(res.estadistico)))
    return all(ok for _, ok, _ in resultados), resultados
