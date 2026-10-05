"""
Subpaquete de generadores de números pseudoaleatorios.

Generadores disponibles:
    - CuadradosMedios: Método de los Cuadrados Medios (Von Neumann, 1946).
    - CongruencialLineal: Congruencial Mixto, x_{i+1} = (a*x_i + c) mod m.
    - CongruencialMultiplicativo: Congruencial con c=0 (Lehmer, 1951).
    - CongruencialAditivo: Fibonacci Retrasado, x_{i+1} = (x_{i-j} + x_{i-k}) mod m.

Transformaciones (en módulo transformaciones):
    - uniforme(ri, a, b):  R_i → N_i ~ U(a, b)
    - normal_box_muller(ri): R_i → N_i ~ N(μ, σ²)
    - exponencial(ri, λ):  R_i → N_i ~ Exp(λ)
"""

from .cuadrados_medios import CuadradosMedios
from .congruencial_lineal import CongruencialLineal
from .congruencial_multiplicativo import CongruencialMultiplicativo
from .congruencial_aditivo import CongruencialAditivo
