"""
Clase estandarizada para el resultado de cualquier prueba estadística.
Permite que todas las pruebas retornen un objeto con la misma interfaz,
facilitando la visualización y comparación de resultados.
"""


class ResultadoPrueba:
    """
    Resultado estandarizado de una prueba estadística.

    Attributes:
        nombre (str): Nombre de la prueba.
        aprobada (bool): True si la secuencia pasó la prueba.
        estadistico (float): Valor del estadístico calculado.
        valor_critico (float): Valor crítico de referencia.
        limite_inferior (float): Límite inferior del intervalo de confianza (si aplica).
        limite_superior (float): Límite superior del intervalo de confianza (si aplica).
        alpha (float): Nivel de significancia utilizado.
        detalles (dict): Diccionario con datos adicionales específicos de cada prueba.
    """

    def __init__(self, nombre, aprobada, estadistico, valor_critico,
                 limite_inferior=None, limite_superior=None, alpha=0.05,
                 detalles=None):
        self.nombre = nombre
        self.aprobada = aprobada
        self.estadistico = estadistico
        self.valor_critico = valor_critico
        self.limite_inferior = limite_inferior
        self.limite_superior = limite_superior
        self.alpha = alpha
        self.detalles = detalles if detalles is not None else {}

    def __str__(self):
        estado = "APROBADA ✓" if self.aprobada else "RECHAZADA x"
        lineas = [
            f"═══════════════════════════════════════════",
            f"  Prueba: {self.nombre}",
            f"  Estado: {estado}",
            f"  Estadístico: {self.estadistico:.6f}",
            f"  Valor crítico: {self.valor_critico:.6f}",
        ]
        if self.limite_inferior is not None:
            lineas.append(f"  Límite inferior: {self.limite_inferior:.6f}")
        if self.limite_superior is not None:
            lineas.append(f"  Límite superior: {self.limite_superior:.6f}")
        lineas.append(f"  α = {self.alpha}")
        lineas.append(f"═══════════════════════════════════════════")
        return "\n".join(lineas)

    def __repr__(self):
        return (f"ResultadoPrueba(nombre='{self.nombre}', "
                f"aprobada={self.aprobada}, "
                f"estadistico={self.estadistico:.6f})")
