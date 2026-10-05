"""
Método Congruencial Aditivo.

Fórmula:  x_n = (x_{n-j} + x_{n-k}) mod m    con k > j ≥ 1

Requiere una secuencia inicial de al menos k semillas (proporcionadas
manualmente o generadas con otro método como el Congruencial Lineal).
"""


class CongruencialAditivo:
    """
    Generador congruencial aditivo.

    Parámetros:
        semillas_iniciales (list): Lista de al menos k semillas enteras.
        j (int): Índice de retardo menor  (j ≥ 1).
        k (int): Índice de retardo mayor  (k > j).
        m (int): Módulo.

    El generador conserva su estado (los últimos k valores): cada llamada a
    generar() o siguiente() continúa la secuencia donde quedó la anterior.
    Use reiniciar() para volver a las semillas iniciales (o cambiarlas).

    Ejemplo:
        >>> # Semillas iniciales generadas manualmente o con otro generador
        >>> semillas = [23, 99, 45, 12, 67, 38, 54, 71, 88, 10]
        >>> gen = CongruencialAditivo(semillas, j=3, k=7, m=100)
        >>> xi, ri = gen.generar(50)
        >>> r = gen.siguiente()  # continúa la secuencia
    """

    def __init__(self, semillas_iniciales, j, k, m):
        if k <= j:
            raise ValueError(f"k ({k}) debe ser mayor que j ({j})")
        if j < 1:
            raise ValueError("j debe ser >= 1")
        if len(semillas_iniciales) < k:
            raise ValueError(
                f"Se necesitan al menos k={k} semillas iniciales, "
                f"se recibieron {len(semillas_iniciales)}"
            )
        if m <= 0:
            raise ValueError("El módulo m debe ser positivo")

        self.semillas_iniciales = list(semillas_iniciales)
        self.j = j
        self.k = k
        self.m = m
        self.nombre = "Congruencial Aditivo"
        # Estado actual: valores previos necesarios para calcular el siguiente
        self._estado = list(self.semillas_iniciales)

    @classmethod
    def desde_lcg(cls, x0, a, c, m_lcg, j, k, m, n_semillas=None):
        """
        Genera las semillas iniciales automáticamente usando un
        generador congruencial lineal.

        Args:
            x0, a, c, m_lcg: Parámetros del LCG para generar semillas.
            j, k, m: Parámetros del generador aditivo.
            n_semillas: Cantidad de semillas a generar (default = k).
        """
        from .congruencial_lineal import CongruencialLineal

        if n_semillas is None:
            n_semillas = k

        lcg = CongruencialLineal(x0, a, c, m_lcg)
        xi_lcg, _ = lcg.generar(n_semillas)
        # Tomar las semillas generadas (excluyendo x0 del LCG)
        semillas = [x % m for x in xi_lcg[1:n_semillas + 1]]

        return cls(semillas, j, k, m)

    def siguiente(self):
        """
        Genera y retorna el siguiente número R_i ∈ [0, 1), avanzando el estado.

        Returns:
            float: Siguiente R_i de la secuencia.
        """
        x_nuevo = (self._estado[-self.j] + self._estado[-self.k]) % self.m
        self._estado.append(x_nuevo)
        # Solo se necesitan los últimos k valores; se descarta el resto
        # para que la memoria no crezca en simulaciones largas.
        if len(self._estado) > self.k:
            del self._estado[:-self.k]
        return x_nuevo / self.m

    def reiniciar(self, semillas_iniciales=None):
        """
        Reinicia el generador a sus semillas iniciales, o a nuevas semillas.

        Args:
            semillas_iniciales (list, opcional): Nueva lista de al menos k semillas.
                                                 Si se omite, se usan las originales.
        """
        if semillas_iniciales is not None:
            if len(semillas_iniciales) < self.k:
                raise ValueError(
                    f"Se necesitan al menos k={self.k} semillas iniciales, "
                    f"se recibieron {len(semillas_iniciales)}"
                )
            self.semillas_iniciales = list(semillas_iniciales)
        self._estado = list(self.semillas_iniciales)

    def generar(self, cantidad):
        """
        Genera una secuencia de números pseudoaleatorios a partir del estado actual.

        Args:
            cantidad (int): Cantidad de números R_i nuevos a generar
                            (sin contar las semillas iniciales).

        Returns:
            tuple: (xi_list, ri_list)
                - xi_list: Estado previo seguido de los generados. En la primera
                           llamada, el estado previo son las semillas iniciales.
                - ri_list: Solo los R_i nuevos generados (len = cantidad).
        """
        xi_list = list(self._estado)
        ri_list = []

        for _ in range(cantidad):
            # x_n = (x_{n-j} + x_{n-k}) mod m
            #   donde n es la posición del nuevo elemento (len(xi_list))
            x_nuevo = (xi_list[-self.j] + xi_list[-self.k]) % self.m
            xi_list.append(x_nuevo)
            ri_list.append(x_nuevo / self.m)

        self._estado = xi_list[-self.k:]
        return xi_list, ri_list

    def __repr__(self):
        return (f"CongruencialAditivo(j={self.j}, k={self.k}, m={self.m}, "
                f"n_semillas={len(self.semillas_iniciales)})")
