"""
Módulo integrado de generación y validación de números pseudoaleatorios.
Implementado desde cero (from scratch) para el curso de Simulación por Computador.
UPTC - Universidad Pedagógica y Tecnológica de Colombia.

Uso rápido:
    from generador_numeros_pruebas.generadores import CongruencialLineal
    from generador_numeros_pruebas.pruebas import PruebaMedias

    gen = CongruencialLineal(x0=7, a=5, c=3, m=1024)
    xi, ri = gen.generar(1000)

    resultado = PruebaMedias(ri).ejecutar()
    print(resultado)
"""

from .generadores import (
    CuadradosMedios,
    CongruencialLineal,
    CongruencialMultiplicativo,
    CongruencialAditivo,
)
from .generadores.transformaciones import uniforme, normal_box_muller, exponencial
from .pruebas import (
    PruebaMedias,
    PruebaVarianza,
    PruebaChi2,
    PruebaKS,
    PruebaPoker,
    ResultadoPrueba,
)

__version__ = "1.0.0"
