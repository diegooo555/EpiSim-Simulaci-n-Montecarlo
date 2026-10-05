"""
Método Congruencial Multiplicativo.
Autor original: Derrick Henry Lehmer (1951).

Fórmula:  x_{i+1} = (a · x_i) mod m     (caso especial con c = 0)

Requisitos:
    - x0 debe ser coprimario con m:  gcd(x0, m) = 1
    - Para período máximo (m - 1) con m primo, a debe ser raíz primitiva mod m.
    - Nunca genera x_i = 0 (ya que c = 0 haría que la secuencia se detenga).

Normalización: R_i = x_i / m   ->  R_i ∈ (0, 1)
"""
import math


class CongruencialMultiplicativo:
    """
    Generador congruencial multiplicativo (c = 0).

    Parámetros:
        x0 (int): Semilla inicial, 1 <= x0 < m, gcd(x0, m) = 1.
        a  (int): Multiplicador, 1 < a < m.
        m  (int): Módulo (típicamente primo o 2^k).
        normalizar_con_m_menos_1 (bool): Si True, R_i = x_i/(m-1).
            ADVERTENCIA: puede producir R_i = 1.0 (fuera de [0, 1)).

    El generador conserva su estado: cada llamada a generar() o siguiente()
    continúa la secuencia donde quedó la anterior. Use reiniciar() para
    volver a la semilla (o cambiarla).

    Ejemplo (MINSTD, Park & Miller):
        >>> gen = CongruencialMultiplicativo(x0=17, a=16807, m=2^31 - 1)
        >>> xi, ri = gen.generar(1000)
        >>> r = gen.siguiente()  # R_1001, continúa la secuencia
    """

    def __init__(self, x0, a, m, normalizar_con_m_menos_1=False):
        if m <= 0:
            raise ValueError("El módulo m debe ser positivo")
        if not (1 <= x0 < m):
            raise ValueError(f"La semilla x0 debe estar en [1, {m - 1}]")
        if math.gcd(x0, m) != 1:
            raise ValueError(
                f"La semilla x0={x0} debe ser coprimaria con m={m} "
                f"(gcd = {math.gcd(x0, m)})"
            )
        if not (1 < a < m):
            raise ValueError(f"El multiplicador a debe estar en (1, {m})")

        self.x0 = x0
        self.a = a
        self.m = m
        self.normalizar_con_m_menos_1 = normalizar_con_m_menos_1
        self.nombre = "Congruencial Multiplicativo"
        self._x = x0  # Estado actual del generador

    def siguiente(self):
        """
        Genera y retorna el siguiente número R_i, avanzando el estado.

        Returns:
            float: Siguiente R_i de la secuencia.
        """
        divisor = (self.m - 1) if self.normalizar_con_m_menos_1 else self.m
        self._x = (self.a * self._x) % self.m
        return self._x / divisor

    def reiniciar(self, x0=None):
        """
        Reinicia el generador a su semilla inicial, o a una nueva semilla.

        Args:
            x0 (int, opcional): Nueva semilla, 1 <= x0 < m y gcd(x0, m) = 1.
                                Si se omite, se usa la original.
        """
        if x0 is not None:
            if not (1 <= x0 < self.m):
                raise ValueError(f"La semilla x0 debe estar en [1, {self.m - 1}]")
            if math.gcd(x0, self.m) != 1:
                raise ValueError(
                    f"La semilla x0={x0} debe ser coprimaria con m={self.m} "
                    f"(gcd = {math.gcd(x0, self.m)})"
                )
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
            x = (self.a * x) % self.m
            xi_list.append(x)
            ri_list.append(x / divisor)

        self._x = x
        return xi_list, ri_list

    def __repr__(self):
        return (f"CongruencialMultiplicativo(x0={self.x0}, a={self.a}, "
                f"m={self.m})")
