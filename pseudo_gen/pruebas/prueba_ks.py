"""
Prueba de Kolmogorov-Smirnov (KS) (Propiedad 2 - Uniformidad).

Compara la Función de Distribución Empírica Acumulada Sn(x) con la
Función de Distribución Teórica F(x) = x para U(0,1).

Estadístico:
    D = max(D+, D-)
    D+ = max_{1 <= i <= N} (i/N - R_(i))
    D- = max_{1 <= i <= N} (R_(i) - (i-1)/N)

Criterio:
    Aprobada si D <= D_critico
"""
import math
import numpy as np
from scipy import stats
from .resultado import ResultadoPrueba


class PruebaKS:
    """
    Prueba de Kolmogorov-Smirnov para uniformidad en [0, 1).
    """

    def __init__(self, ri_nums, alpha=0.05):
        if not ri_nums:
            raise ValueError("La secuencia ri_nums no puede estar vacía")
        self.ri = list(ri_nums)
        self.n = len(self.ri)
        self.alpha = alpha

    def calcular_critico(self):
        """Calcula el valor crítico de KS según el tamaño muestral."""
        if self.n <= 50:
            return float(stats.ksone.ppf(1.0 - self.alpha / 2.0, self.n))
        else:
            return float(stats.kstwobign.isf(self.alpha) / math.sqrt(self.n))

    def ejecutar(self):
        """
        Ejecuta la prueba de Kolmogorov-Smirnov.

        Returns:
            ResultadoPrueba: Objeto con el veredicto y estadísticas detalladas.
        """
        ordenados = sorted(self.ri)
        n = self.n

        d_plus_list = []
        d_minus_list = []
        d_list = []

        fec_y = []       # Sn(x) = i / n
        teorica_y = []   # F(x) = x para U(0,1)

        d_max = 0.0
        idx_max = 0

        for i, val in enumerate(ordenados, start=1):
            sn = i / n
            sn_anterior = (i - 1) / n
            fx = val  # F(x) = x para U(0, 1)

            dp = sn - fx
            dm = fx - sn_anterior

            d_plus_list.append(dp)
            d_minus_list.append(dm)
            actual_d = max(abs(dp), abs(dm))
            d_list.append(actual_d)

            fec_y.append(sn)
            teorica_y.append(fx)

            if actual_d > d_max:
                d_max = actual_d
                idx_max = i - 1

        d_critico = self.calcular_critico()
        aprobada = d_max <= d_critico

        detalles = {
            "n": n,
            "numeros_ordenados": ordenados,
            "fec_y": fec_y,
            "teorica_y": teorica_y,
            "d_plus": d_plus_list,
            "d_minus": d_minus_list,
            "d_max": d_max,
            "d_critico": d_critico,
            "punto_d_max": (ordenados[idx_max], fec_y[idx_max]),
        }

        return ResultadoPrueba(
            nombre="Prueba Kolmogorov-Smirnov",
            aprobada=aprobada,
            estadistico=d_max,
            valor_critico=d_critico,
            limite_inferior=None,
            limite_superior=d_critico,
            alpha=self.alpha,
            detalles=detalles
        )
