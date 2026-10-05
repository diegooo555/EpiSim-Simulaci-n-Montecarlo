"""
Subpaquete de pruebas estadísticas para validación de secuencias pseudoaleatorias.

Pruebas disponibles (las 5 requeridas por el taller):
    - PruebaMedias:   Propiedad I  – Aleatoriedad (E[Ri] = 0.5).
    - PruebaVarianza:  Propiedad I  – Aleatoriedad (Var(Ri) = 1/12).
    - PruebaChi2:      Propiedad II – Uniformidad (χ² sobre [0,1)).
    - PruebaKS:        Propiedad II – Uniformidad (Kolmogorov-Smirnov).
    - PruebaPoker:     Propiedad III – Independencia serial (patrones de 5 dígitos).

Todas operan exclusivamente sobre el conjunto Ri (uniformes normalizados en [0, 1)).
"""

from .resultado import ResultadoPrueba
from .prueba_medias import PruebaMedias
from .prueba_varianza import PruebaVarianza
from .prueba_chi2 import PruebaChi2
from .prueba_ks import PruebaKS
from .prueba_poker import PruebaPoker
