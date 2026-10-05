"""
Método de los Cuadrados Medios (Middle-Square Method).
Autor original: John von Neumann (1946).

Algoritmo:
    1. Tomar semilla x_i de n dígitos.
    2. Calcular x_i^2 (produce hasta 2n dígitos).
    3. Rellenar con ceros a la izquierda hasta 2n dígitos.
    4. Extraer los n dígitos centrales -> x_{i+1}.
    5. Normalizar: R_i = x_{i+1} / 10^n.

Limitaciones conocidas:
    - Puede converger a cero si la semilla llega a 0000...
    - Período corto comparado con métodos congruenciales.
    - La calidad depende fuertemente de la semilla.
"""


class CuadradosMedios:
    """
    Generador de números pseudoaleatorios por Cuadrados Medios.

    Parámetros:
        x0 (int): Semilla inicial.
        n_digitos (int): Cantidad de dígitos de la semilla (típicamente 4, 6, 8 o 10).

    El generador conserva su estado: cada llamada a generar() o siguiente()
    continúa la secuencia donde quedó la anterior. Use reiniciar() para
    volver a la semilla (o cambiarla).

    Ejemplo:
        >>> gen = CuadradosMedios(x0=2222, n_digitos=4)
        >>> xi, ri = gen.generar(3)
        >>> print(ri)  # [0.9372, 0.8343, 0.6056]
        >>> gen.siguiente()  # 0.6751 (continúa la secuencia)
    """

    def __init__(self, x0, n_digitos=4):
        if n_digitos < 1:
            raise ValueError("n_digitos debe ser >= 1")
        if x0 < 0:
            raise ValueError("La semilla debe ser no negativa")

        self.x0 = x0
        self.n_digitos = n_digitos
        self.nombre = "Cuadrados Medios"
        self._x = x0  # Estado actual del generador

    def _siguiente_x(self, x):
        """Aplica un paso del algoritmo: eleva al cuadrado y extrae los n dígitos centrales."""
        n = self.n_digitos
        # Paso 1: Convertir a cadena y rellenar a 2n dígitos
        x_str = str(x * x).zfill(2 * n)
        # Paso 2: Extraer los n dígitos centrales
        #   Ej: n=4, 2n=8 -> inicio=2, "ABCDEFGH"[2:6] = "CDEF"
        inicio = (len(x_str) - n) // 2
        return int(x_str[inicio:inicio + n])

    def siguiente(self):
        """
        Genera y retorna el siguiente número R_i ∈ [0, 1), avanzando el estado.

        Returns:
            float: Siguiente R_i de la secuencia.
        """
        self._x = self._siguiente_x(self._x)
        return self._x / (10 ** self.n_digitos)

    def reiniciar(self, x0=None):
        """
        Reinicia el generador a su semilla inicial, o a una nueva semilla.

        Args:
            x0 (int, opcional): Nueva semilla. Si se omite, se usa la original.
        """
        if x0 is not None:
            if x0 < 0:
                raise ValueError("La semilla debe ser no negativa")
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
                - ri_list: Números uniformes R_i ∈ [0, 1) (len = cantidad).
        """
        divisor = 10 ** self.n_digitos

        xi_list = [self._x]
        ri_list = []

        for _ in range(cantidad):
            self._x = self._siguiente_x(self._x)
            xi_list.append(self._x)
            ri_list.append(self._x / divisor)

        return xi_list, ri_list

    def detectar_ciclo(self, max_iter=10000):
        """
        Detecta si el generador entra en un ciclo, partiendo de x0.
        No modifica el estado actual del generador.

        Returns:
            tuple: (longitud_ciclo, inicio_ciclo) o (None, None) si no se detecta.
        """
        vistos = {}
        x = self.x0

        for i in range(max_iter):
            if x in vistos:
                return (i - vistos[x], vistos[x])
            vistos[x] = i
            x = self._siguiente_x(x)

        return (None, None)

    def __repr__(self):
        return (f"CuadradosMedios(x0={self.x0}, n_digitos={self.n_digitos})")
