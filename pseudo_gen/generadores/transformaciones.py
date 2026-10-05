"""
Transformaciones de distribución.

Funciones que transforman una secuencia de R_i ~ U(0, 1)
a otras distribuciones de probabilidad.

Transformaciones implementadas:
    - uniforme(ri, a, b):       R_i -> N_i ~ U(a, b)
    - normal_box_muller(ri):    R_i -> Z_i ~ N(μ, σ²)   (Box-Muller)
    - exponencial(ri, λ):       R_i -> N_i ~ Exp(λ)      (Inversa)

NOTA: No se usa numpy.random ni random. Solo math para operaciones
      aritméticas básicas (sqrt, log, cos, sin, pi).
"""
import math


def uniforme(ri_list, a, b):
    """
    Transforma R_i ~ U(0,1)  ->  N_i ~ U(a, b).

    Fórmula:  N_i = a + (b - a) * R_i

    Args:
        ri_list (list[float]): Secuencia de R_i ∈ [0, 1).
        a (float): Límite inferior del intervalo.
        b (float): Límite superior del intervalo.

    Returns:
        list[float]: Secuencia N_i ∈ [a, b).
    """
    if b <= a:
        raise ValueError(f"b ({b}) debe ser mayor que a ({a})")
    rango = b - a
    return [a + rango * r for r in ri_list]


def normal_box_muller(ri_list, mu=0.0, sigma=1.0):
    """
    Transforma pares de R_i ~ U(0,1)  ->  Z_i ~ N(μ, σ²)
    usando el método de Box-Muller.

    Fórmulas:
        Z₁ = √(-2*ln(R₁)) * cos(2π*R₂)
        Z₂ = √(-2*ln(R₁)) * sin(2π*R₂)
        N_i = μ + σ*Z_i

    NOTA: Consume pares de R_i. Si len(ri_list) es impar,
          el último valor se descarta.

    Args:
        ri_list (list[float]): Secuencia de R_i ∈ [0, 1).
        mu (float): Media de la distribución normal destino.
        sigma (float): Desviación estándar de la distribución normal destino.

    Returns:
        list[float]: Secuencia de valores normales.
    """
    resultado = []
    for i in range(0, len(ri_list) - 1, 2):
        r1 = ri_list[i]
        r2 = ri_list[i + 1]

        # Protección contra log(0)
        if r1 <= 0:
            r1 = 1e-10

        factor = math.sqrt(-2.0 * math.log(r1))
        angulo = 2.0 * math.pi * r2

        z1 = factor * math.cos(angulo)
        z2 = factor * math.sin(angulo)

        resultado.append(mu + sigma * z1)
        resultado.append(mu + sigma * z2)

    return resultado


def exponencial(ri_list, lambd=1.0):
    """
    Transforma R_i ~ U(0,1)  ->  N_i ~ Exp(λ)
    usando la transformación inversa.

    Fórmula:  N_i = -(1/λ) * ln(R_i)

    Args:
        ri_list (list[float]): Secuencia de R_i ∈ [0, 1).
        lambd (float): Parámetro de tasa (λ > 0).

    Returns:
        list[float]: Secuencia de valores exponenciales.
    """
    if lambd <= 0:
        raise ValueError("lambda debe ser positivo")
    inv_lambd = 1.0 / lambd
    resultado = []
    for r in ri_list:
        if r <= 0:
            r = 1e-10
        resultado.append(-inv_lambd * math.log(r))
    return resultado
