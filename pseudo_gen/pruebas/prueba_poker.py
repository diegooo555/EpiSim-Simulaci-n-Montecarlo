"""
Prueba de Póker (Propiedad 3 - Independencia Serial).

Evalúa la independencia serial analizando las secuencias de 5 dígitos decimales
de cada número R_i, clasificándolas en 7 manos de póker:

    1. Todos diferentes (D):          P = 0.3024
    2. Un par (O):                    P = 0.5040
    3. Dos pares (T):                 P = 0.1080
    4. Tercia (K):                    P = 0.0720
    5. Tercia y par / Full house (F): P = 0.0090
    6. Cuatro iguales / Póker (P):    P = 0.0045
    7. Todos iguales / Quintilla (Q): P = 0.0001

Estadístico:
    χ₀^2 = ∑ ((Oi - Ei)^2 / Ei) con 6 grados de libertad.
"""
from collections import Counter
from scipy.stats import chi2
from .resultado import ResultadoPrueba


class PruebaPoker:
    """
    Prueba de Póker para secuencias de 5 dígitos decimales.
    """

    CATEGORIAS = [
        "Todos Diferentes (D)",
        "Un Par (O)",
        "Dos Pares (T)",
        "Tercia (K)",
        "Full House (F)",
        "Póker (P)",
        "Quintilla (Q)"
    ]

    PROBABILIDADES = [0.3024, 0.5040, 0.1080, 0.0720, 0.0090, 0.0045, 0.0001]

    def __init__(self, ri_nums, alpha=0.05, n_digitos=5):
        if not ri_nums:
            raise ValueError("La secuencia ri_nums no puede estar vacía")
        self.ri = list(ri_nums)
        self.n = len(self.ri)
        self.alpha = alpha
        self.n_digitos = n_digitos

    def clasificar_mano(self, cadena_digitos):
        """
        Clasifica una cadena de dígitos en una de las 7 manos de póker.

        Returns:
            int: Índice de la categoría (0 a 6).
        """
        conteos = sorted(Counter(cadena_digitos).values(), reverse=True)

        if conteos[0] == 5:
            return 6  # Quintilla
        elif conteos[0] == 4:
            return 5  # Póker
        elif conteos[0] == 3 and len(conteos) > 1 and conteos[1] == 2:
            return 4  # Full House
        elif conteos[0] == 3:
            return 3  # Tercia
        elif conteos[0] == 2 and len(conteos) > 1 and conteos[1] == 2:
            return 2  # Dos Pares
        elif conteos[0] == 2:
            return 1  # Un Par
        else:
            return 0  # Todos Diferentes

    def ejecutar(self):
        """
        Ejecuta la prueba de Póker.

        Returns:
            ResultadoPrueba: Objeto con el veredicto y estadísticas detalladas.
        """
        frec_observadas = [0] * 7

        for r in self.ri:
            # Obtener representación de n_digitos decimales
            partes = f"{r:.10f}".split(".")
            digitos = partes[1][:self.n_digitos] if len(partes) > 1 else "0" * self.n_digitos
            if len(digitos) < self.n_digitos:
                digitos = digitos.ljust(self.n_digitos, "0")

            idx_mano = self.clasificar_mano(digitos)
            frec_observadas[idx_mano] += 1

        frec_esperadas = [p * self.n for p in self.PROBABILIDADES]

        terminos_chi2 = []
        for oi, ei in zip(frec_observadas, frec_esperadas):
            if ei > 0:
                terminos_chi2.append(((oi - ei) ** 2) / ei)
            else:
                terminos_chi2.append(0.0)

        chi_calculado = sum(terminos_chi2)
        gl = len(self.CATEGORIAS) - 1
        chi_critico = chi2.ppf(1.0 - self.alpha, gl)

        aprobada = chi_calculado <= chi_critico

        detalles = {
            "n": self.n,
            "categorias": self.CATEGORIAS,
            "probabilidades": self.PROBABILIDADES,
            "frecuencias_observadas": frec_observadas,
            "frecuencias_esperadas": frec_esperadas,
            "terminos_chi2": terminos_chi2,
            "chi2_calculado": chi_calculado,
            "chi2_critico": chi_critico,
            "grados_libertad": gl,
        }

        return ResultadoPrueba(
            nombre="Prueba de Póker",
            aprobada=aprobada,
            estadistico=chi_calculado,
            valor_critico=chi_critico,
            limite_inferior=None,
            limite_superior=chi_critico,
            alpha=self.alpha,
            detalles=detalles
        )
