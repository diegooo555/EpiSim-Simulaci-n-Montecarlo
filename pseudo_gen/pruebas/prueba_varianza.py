"""
Prueba de Varianza (Propiedad 1 - Aleatoriedad).

Evalúa si la varianza de la secuencia generada es estadísticamente equivalente
a la varianza teórica de una distribución uniforme continua U(0,1), que es 1/12 ≈ 0.083333.

Hipótesis:
    H0: σ^2 = 1/12
    H1: σ^2 ≠ 1/12

Intervalo de aceptación:
    LI = χ^2_(α/2, n-1) / (12 * (n - 1))
    LS = χ^2_(1 - α/2, n-1) / (12 * (n - 1))
"""
from scipy.stats import chi2
from .resultado import ResultadoPrueba


class PruebaVarianza:
    """
    Prueba de varianza para una secuencia de números pseudoaleatorios R_i ∈ [0, 1).
    """

    def __init__(self, ri_nums, alpha=0.05):
        if len(ri_nums) < 2:
            raise ValueError("Se requieren al menos 2 números para calcular la varianza")
        self.ri = list(ri_nums)
        self.n = len(self.ri)
        self.alpha = alpha

    def ejecutar(self):
        """
        Ejecuta la prueba de varianza.

        Returns:
            ResultadoPrueba: Objeto con el veredicto y estadísticas calculadas.
        """
        promedio = sum(self.ri) / self.n
        # Varianza muestral insesgada con N - 1 grados de libertad
        varianza = sum((x - promedio) ** 2 for x in self.ri) / (self.n - 1)
        gl = self.n - 1

        chi1 = chi2.ppf(self.alpha / 2.0, gl)
        chi2_val = chi2.ppf(1.0 - (self.alpha / 2.0), gl)

        lim_inf = chi1 / (12.0 * gl)
        lim_sup = chi2_val / (12.0 * gl)

        aprobada = lim_inf <= varianza <= lim_sup

        detalles = {
            "n": self.n,
            "grados_libertad": gl,
            "media_calculada": promedio,
            "varianza_calculada": varianza,
            "varianza_esperada": 1.0 / 12.0,
            "chi2_inf": chi1,
            "chi2_sup": chi2_val,
            "limite_inferior": lim_inf,
            "limite_superior": lim_sup,
        }

        return ResultadoPrueba(
            nombre="Prueba de Varianza",
            aprobada=aprobada,
            estadistico=varianza,
            valor_critico=1.0 / 12.0,
            limite_inferior=lim_inf,
            limite_superior=lim_sup,
            alpha=self.alpha,
            detalles=detalles
        )
