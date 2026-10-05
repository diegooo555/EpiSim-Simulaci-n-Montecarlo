"""
Método Congruencial Lineal (Mixto).
Autores: Lehmer (1951), Thomson & Rotenberg (1958).

Fórmula:  x_{i+1} = (a · x_i + c) mod m

Condiciones del Teorema de Hull-Dobell para período completo m:
    1. gcd(c, m) = 1
    2. (a - 1) es divisible por todos los factores primos de m.
    3. Si m es múltiplo de 4, (a - 1) también es múltiplo de 4.

Normalización: R_i = x_i / m   →  R_i ∈ [0, 1)
"""
import math


class CongruencialLineal:
    """
    Generador congruencial lineal (mixto, c != 0).

    Parámetros:
        x0 (int): Semilla inicial, 0 <= x0 < m.
        a  (int): Multiplicador, 0 < a < m.
        c  (int): Incremento, 0 <= c < m (idealmente gcd(c, m) = 1).
        m  (int): Módulo (típicamente potencia de 2).
        normalizar_con_m_menos_1 (bool): Si True, R_i = x_i/(m-1).
            ADVERTENCIA: puede producir R_i = 1.0 (fuera de [0, 1)).

    El generador conserva su estado: cada llamada a generar() o siguiente()
    continúa la secuencia donde quedó la anterior. Use reiniciar() para
    volver a la semilla (o cambiarla).

    Ejemplo:
        >>> gen = CongruencialLineal(x0=12345, a=1664525, c=1013904223, m=2^32)
        >>> xi, ri = gen.generar(10)
        >>> r = gen.siguiente()  # R_11, continúa la secuencia
    """

    def __init__(self, x0, a, c, m, normalizar_con_m_menos_1=False):
        if m <= 0:
            raise ValueError("El módulo m debe ser positivo")
        if not (0 <= x0 < m):
            raise ValueError(f"La semilla x0 debe estar en [0, {m - 1}]")
        if not (0 < a < m):
            raise ValueError(f"El multiplicador a debe estar en (0, {m})")
        if not (0 <= c < m):
            raise ValueError(f"El incremento c debe estar en [0, {m})")

        self.x0 = x0
        self.a = a
        self.c = c
        self.m = m
        self.normalizar_con_m_menos_1 = normalizar_con_m_menos_1
        self.nombre = "Congruencial Lineal"
        self._x = x0  # Estado actual del generador

    # ── Constructores alternativos ──────────────────────────────────────

    @classmethod
    def desde_parametros_clase(cls, x0, k, c, g,
                               normalizar_con_m_menos_1=False):
        """
        Constructor usando la parametrización vista en clase.

        a = 1 + 2k,  m = 2^g

        Args:
            x0: Semilla.
            k:  Parámetro entero para calcular a.
            c:  Incremento.
            g:  Exponente para calcular m.
        """
        a = 1 + 2 * k
        m = 2 ** g
        return cls(x0, a, c, m, normalizar_con_m_menos_1)

    # ── Validación ──────────────────────────────────────────────────────

    def validar_hull_dobell(self):
        """
        Verifica las 3 condiciones del Teorema de Hull-Dobell.

        Returns:
            tuple: (cumple_todo, detalles_dict)
        """
        detalles = {}

        # 1) gcd(c, m) = 1
        cond1 = math.gcd(self.c, self.m) == 1
        detalles['gcd(c, m) = 1'] = cond1

        # 2) (a-1) divisible por cada factor primo de m
        factores = self._factores_primos(self.m)
        a_menos_1 = self.a - 1
        cond2 = all((a_menos_1 % p) == 0 for p in factores)
        detalles['(a-1) div. factores primos de m'] = cond2

        # 3) Si 4 | m → 4 | (a-1)
        if self.m % 4 == 0:
            cond3 = (a_menos_1 % 4) == 0
        else:
            cond3 = True  # No aplica
        detalles['Si 4|m → 4|(a-1)'] = cond3

        return (cond1 and cond2 and cond3), detalles

    # ── Generación ──────────────────────────────────────────────────────

    def siguiente(self):
        """
        Genera y retorna el siguiente número R_i, avanzando el estado.

        Returns:
            float: Siguiente R_i de la secuencia.
        """
        divisor = (self.m - 1) if self.normalizar_con_m_menos_1 else self.m
        self._x = (self.a * self._x + self.c) % self.m
        return self._x / divisor if divisor > 0 else 0.0

    def reiniciar(self, x0=None):
        """
        Reinicia el generador a su semilla inicial, o a una nueva semilla.

        Args:
            x0 (int, opcional): Nueva semilla, 0 <= x0 < m. Si se omite, se usa la original.
        """
        if x0 is not None:
            if not (0 <= x0 < self.m):
                raise ValueError(f"La semilla x0 debe estar en [0, {self.m - 1}]")
            self.x0 = x0
        self._x = self.x0

    def generar(self, cantidad):
        """
        Genera una secuencia de números pseudoaleatorios a partir del estado actual.

        Args:
            cantidad (int): Cantidad de números R_i a generar.

        Returns:
            tuple: (xi_list, ri_list)
                - xi_list: Estado inicial seguido de los enteros generados
                           (len = cantidad + 1). En la primera llamada, xi_list[0] = x0.
                - ri_list: Números R_i (len = cantidad).
        """
        divisor = (self.m - 1) if self.normalizar_con_m_menos_1 else self.m

        xi_list = [self._x]
        ri_list = []

        x = self._x
        for _ in range(cantidad):
            x = (self.a * x + self.c) % self.m
            xi_list.append(x)
            ri_list.append(x / divisor if divisor > 0 else 0.0)

        self._x = x
        return xi_list, ri_list

    # ── Utilidades internas ─────────────────────────────────────────────

    @staticmethod
    def _factores_primos(n):
        """Obtiene el conjunto de factores primos de n."""
        factores = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factores.add(d)
                n //= d
            d += 1
        if n > 1:
            factores.add(n)
        return factores

    def __repr__(self):
        return (f"CongruencialLineal(x0={self.x0}, a={self.a}, "
                f"c={self.c}, m={self.m})")
