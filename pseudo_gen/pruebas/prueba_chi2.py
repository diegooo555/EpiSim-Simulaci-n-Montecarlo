"""
Prueba Chi-Cuadrado de Bondad de Ajuste (Propiedad 2 - Uniformidad).

Verifica si los números R_i se distribuyen uniformemente en el intervalo [0, 1).
Divide el rango [0, 1) en k subintervalos de igual ancho.

Hipótesis:
    H0: Ri ~ U(0, 1)
    H1: Ri no sigue una distribución uniforme

Estadístico:
    χ₀^2 = ∑ ((Oi - Ei)^2 / Ei)
donde:
    Oi = Frecuencia observada en el intervalo i
    Ei = Frecuencia esperada = N / k
    gl = k - 1
Criterio:
    Aprobada si χ₀^2 <= χ^2_(1-α, k-1)
"""
from scipy.stats import chi2
from .resultado import ResultadoPrueba


class PruebaChi2:
    """
    Prueba Chi-cuadrado de uniformidad para R_i ∈ [0, 1).
    """

    def __init__(self, ri_nums, k_intervalos=10, alpha=0.05, rango=(0.0, 1.0)):
        if not ri_nums:
            raise ValueError("La secuencia ri_nums no puede estar vacía")
        if k_intervalos < 2:
            raise ValueError("El número de intervalos k debe ser >= 2")

        self.ri = list(ri_nums)
        self.n = len(self.ri)
        self.k = k_intervalos
        self.alpha = alpha
        self.rango_min, self.rango_max = rango

    def ejecutar(self):
        """
        Ejecuta la prueba Chi-cuadrado.

        Returns:
            ResultadoPrueba: Objeto con el veredicto y estadísticas detalladas.
        """
        ancho = (self.rango_max - self.rango_min) / self.k
        limites = [self.rango_min + i * ancho for i in range(self.k + 1)]
        frec_esperada = self.n / self.k

        frec_observada = [0] * self.k
        for x in self.ri:
            # Encontrar el intervalo correspondiente
            if x < self.rango_min or x > self.rango_max:
                continue
            idx = int((x - self.rango_min) / ancho)
            if idx >= self.k:
                idx = self.k - 1  # Límite superior inclusivo
            frec_observada[idx] += 1

        # Cálculo de estadístico Chi2 por intervalo
        chi_terminos = []
        for oi in frec_observada:
            termino = ((oi - frec_esperada) ** 2) / frec_esperada
            chi_terminos.append(termino)

        chi_calculado = sum(chi_terminos)
        gl = self.k - 1
        chi_critico = chi2.ppf(1.0 - self.alpha, gl)

        aprobada = chi_calculado <= chi_critico

        detalles = {
            "n": self.n,
            "k_intervalos": self.k,
            "grados_libertad": gl,
            "limites_intervalos": limites,
            "frecuencias_observadas": frec_observada,
            "frecuencia_esperada": frec_esperada,
            "terminos_chi2": chi_terminos,
            "chi2_calculado": chi_calculado,
            "chi2_critico": chi_critico,
        }

        return ResultadoPrueba(
            nombre="Prueba Chi-Cuadrado",
            aprobada=aprobada,
            estadistico=chi_calculado,
            valor_critico=chi_critico,
            limite_inferior=None,
            limite_superior=chi_critico,
            alpha=self.alpha,
            detalles=detalles
        )
