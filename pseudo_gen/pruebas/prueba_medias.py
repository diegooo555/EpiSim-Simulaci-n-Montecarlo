"""
Prueba de Medias (Propiedad 1 - Aleatoriedad).

Evalúa si los números de la secuencia tienen un valor esperado cercano a 0.5.
Hipótesis:
    H0: μ = 0.5
    H1: μ != 0.5

Estadístico:
    Z = (X̄ - 0.5) * √(12 * N)
Intervalo de aceptación:
    LI = 0.5 - Z_(α/2) / √(12 * N)
    LS = 0.5 + Z_(α/2) / √(12 * N)
"""
import math
from scipy.stats import norm
from .resultado import ResultadoPrueba


class PruebaMedias:
    """
    Prueba de medias para una secuencia de números pseudoaleatorios R_i ∈ [0, 1).
    """

    def __init__(self, ri_nums, alpha=0.05):
        if not ri_nums:
            raise ValueError("La secuencia ri_nums no puede estar vacía")
        self.ri = list(ri_nums)
        self.n = len(self.ri)
        self.alpha = alpha

    def ejecutar(self):
        """
        Ejecuta la prueba de medias.

        Returns:
            ResultadoPrueba: Objeto con el veredicto y estadísticas calculadas.
        """
        promedio = sum(self.ri) / self.n
        z_critico = norm.ppf(1.0 - (self.alpha / 2.0))
        error_estandar = 1.0 / math.sqrt(12.0 * self.n)

        lim_inf = 0.5 - (z_critico * error_estandar)
        lim_sup = 0.5 + (z_critico * error_estandar)

        z_calc = (promedio - 0.5) / error_estandar
        aprobada = lim_inf <= promedio <= lim_sup

        detalles = {
            "n": self.n,
            "media_calculada": promedio,
            "media_esperada": 0.5,
            "z_calculado": z_calc,
            "z_critico": z_critico,
            "error_estandar": error_estandar,
            "limite_inferior": lim_inf,
            "limite_superior": lim_sup,
        }

        return ResultadoPrueba(
            nombre="Prueba de Medias",
            aprobada=aprobada,
            estadistico=promedio,
            valor_critico=0.5,
            limite_inferior=lim_inf,
            limite_superior=lim_sup,
            alpha=self.alpha,
            detalles=detalles
        )
