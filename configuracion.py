"""
Carga y validación de los parámetros de un escenario de EpiSim.

Los parámetros se leen desde un archivo plano, en cualquiera de estos dos formatos:

    .csv  →  parametro,valor,descripcion
             beta,"U(0.3, 0.5)",Contactos promedio por día
             poblacion,10000,Individuos

    .txt  →  beta = U(0.3, 0.5)     # comentario
             poblacion = 10000

Los parámetros estocásticos se escriben como U(a, b). Un valor fijo v equivale a U(v, v).
Cualquier parámetro que no aparezca en el archivo toma el valor por defecto de la tabla
PARAMETROS_POR_DEFECTO (los valores del enunciado del taller).
"""
import copy
import csv
import os
import re

# Parámetros que se muestrean con una distribución U(a, b).
#   - Por simulación: beta, p_transmision, cobertura_vacunacion, efectividad_vacuna, letalidad
#   - Por individuo (al entrar al estado): periodo_incubacion, periodo_infeccioso (en días enteros)
PARAMETROS_UNIFORMES = {
    "beta": (0.3, 0.5),                  # Contactos promedio por día de cada infectado
    "periodo_incubacion": (2, 5),        # Días en E
    "periodo_infeccioso": (7, 14),       # Días en I
    "p_transmision": (0.4, 0.6),         # Prob. de contagio si hay contacto
    "cobertura_vacunacion": (0.3, 0.7),  # Proporción de la población vacunada
    "efectividad_vacuna": (0.80, 0.95),  # Reducción de susceptibilidad del vacunado
    "letalidad": (0.005, 0.015),         # SUPUESTO: prob. de fallecer al salir de I
}

PARAMETROS_ENTEROS = {
    "poblacion": 10_000,
    "infectados_iniciales": 10,
    "dias": 365,
    "n_simulaciones": 1000,
    "umbral_control": 0,        # Brote controlado cuando E + I <= umbral (después del pico)
    "semilla_maestra": 12345,
}

PARAMETROS_REALES = {
    "alpha": 0.05,              # Nivel de significancia de las pruebas de los R_i
}

PARAMETROS_OPCION = {
    "modo_contactos": ("estocastico", "literal"),
    "seleccion_contacto": ("poblacion", "susceptibles"),
    "generador": ("lcg", "mult"),
    "validar_numeros": ("si", "no"),
}

PARAMETROS_TEXTO = {
    "nombre": "escenario",
    "archivo_semillas": "",     # Opcional: semillas explícitas en lugar de la semilla maestra
}

PARAMETROS_POR_DEFECTO = {
    **PARAMETROS_UNIFORMES,
    **PARAMETROS_ENTEROS,
    **PARAMETROS_REALES,
    **{k: v[0] for k, v in PARAMETROS_OPCION.items()},
    **PARAMETROS_TEXTO,
}

# Parámetros que se muestrean una vez al inicio de cada simulación (en este orden)
PARAMETROS_POR_SIMULACION = ["beta", "p_transmision", "cobertura_vacunacion",
                             "efectividad_vacuna", "letalidad"]

_PROBABILIDADES = ["p_transmision", "cobertura_vacunacion", "efectividad_vacuna", "letalidad"]
_PERIODOS = ["periodo_incubacion", "periodo_infeccioso"]

_PATRON_UNIFORME = re.compile(r"^U\(\s*([^,;]+?)\s*[,;]\s*([^)]+?)\s*\)$", re.IGNORECASE)


class Configuracion:
    """
    Conjunto de parámetros de un escenario.

    Se accede como un diccionario: cfg["beta"] → (0.3, 0.5), cfg["poblacion"] → 10000.
    """

    def __init__(self, valores=None, origen="(valores por defecto)"):
        self.valores = copy.deepcopy(PARAMETROS_POR_DEFECTO)
        if valores:
            for clave, valor in valores.items():
                self.valores[clave] = _convertir(clave, valor)
        self.origen = origen
        self.validar()

    def __getitem__(self, clave):
        return self.valores[clave]

    def copia(self, **cambios):
        """Retorna una copia con algunos parámetros modificados (p. ej. para sensibilidad)."""
        nueva = copy.deepcopy(self)
        for clave, valor in cambios.items():
            nueva.valores[clave] = _convertir(clave, valor)
        nueva.validar()
        return nueva

    def validar(self):
        v = self.valores
        for clave in PARAMETROS_UNIFORMES:
            a, b = v[clave]
            if a > b:
                raise ValueError(f"{clave}: el límite inferior ({a}) es mayor que el superior ({b})")
        for clave in _PROBABILIDADES:
            a, b = v[clave]
            if not (0 <= a <= 1 and 0 <= b <= 1):
                raise ValueError(f"{clave} debe estar en [0, 1]")
        for clave in _PERIODOS:
            a, b = v[clave]
            if a < 1 or a != int(a) or b != int(b):
                raise ValueError(f"{clave} debe ser un rango de días enteros >= 1")
            v[clave] = (int(a), int(b))
        if v["beta"][0] < 0:
            raise ValueError("beta debe ser >= 0")
        if v["poblacion"] < 1 or not (0 < v["infectados_iniciales"] <= v["poblacion"]):
            raise ValueError("Se requiere poblacion >= 1 y 0 < infectados_iniciales <= poblacion")
        if v["dias"] < 1 or v["n_simulaciones"] < 1:
            raise ValueError("dias y n_simulaciones deben ser >= 1")
        if not (0 < v["alpha"] < 1):
            raise ValueError("alpha debe estar en (0, 1)")

    def tabla(self):
        """Filas (parametro, valor legible) para imprimir o guardar."""
        filas = []
        for clave, valor in self.valores.items():
            if clave in PARAMETROS_UNIFORMES:
                a, b = valor
                valor = f"{a}" if a == b else f"U({a}, {b})"
            filas.append((clave, valor))
        return filas

    def guardar(self, ruta):
        """Guarda la configuración efectiva en .csv (formato de lectura de cargar_configuracion)."""
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["parametro", "valor"])
            w.writerows(self.tabla())


def cargar_configuracion(ruta):
    """
    Lee un archivo de parámetros (.csv o .txt).

    Args:
        ruta (str): Ruta del archivo.

    Returns:
        Configuracion
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {ruta}")

    extension = os.path.splitext(ruta)[1].lower()
    valores = {}
    with open(ruta, "r", encoding="utf-8-sig") as f:
        if extension == ".csv":
            for fila in csv.reader(f):
                if not fila or not fila[0].strip() or fila[0].strip().startswith("#"):
                    continue
                clave = fila[0].strip().lower()
                if clave == "parametro":        # Encabezado
                    continue
                valores[clave] = fila[1].strip() if len(fila) > 1 else ""
        else:
            for linea in f:
                linea = linea.split("#", 1)[0].strip()
                if not linea:
                    continue
                if "=" not in linea:
                    raise ValueError(f"Línea sin '=' en {ruta}: {linea}")
                clave, valor = linea.split("=", 1)
                valores[clave.strip().lower()] = valor.strip()

    desconocidos = set(valores) - set(PARAMETROS_POR_DEFECTO)
    if desconocidos:
        raise ValueError(f"Parámetros desconocidos en {ruta}: {', '.join(sorted(desconocidos))}")

    cfg = Configuracion(valores, origen=ruta)
    if cfg["nombre"] == PARAMETROS_TEXTO["nombre"]:
        cfg.valores["nombre"] = os.path.splitext(os.path.basename(ruta))[0]
    return cfg


def _convertir(clave, valor):
    """Convierte el texto leído del archivo al tipo que corresponde a cada parámetro."""
    if clave not in PARAMETROS_POR_DEFECTO:
        raise ValueError(f"Parámetro desconocido: {clave}")

    if clave in PARAMETROS_UNIFORMES:
        if isinstance(valor, (tuple, list)):
            a, b = valor
        elif isinstance(valor, (int, float)):
            a = b = valor
        else:
            texto = str(valor).strip()
            m = _PATRON_UNIFORME.match(texto)
            if m:
                a, b = m.group(1), m.group(2)
            else:
                a = b = texto
        try:
            return (float(a), float(b))
        except ValueError:
            raise ValueError(f"{clave}: se esperaba un número o U(a, b), se leyó '{valor}'")

    if clave in PARAMETROS_ENTEROS:
        try:
            return int(float(valor))
        except ValueError:
            raise ValueError(f"{clave}: se esperaba un entero, se leyó '{valor}'")

    if clave in PARAMETROS_REALES:
        try:
            return float(valor)
        except ValueError:
            raise ValueError(f"{clave}: se esperaba un número, se leyó '{valor}'")

    if clave in PARAMETROS_OPCION:
        texto = str(valor).strip().lower()
        if texto not in PARAMETROS_OPCION[clave]:
            raise ValueError(f"{clave}: opciones válidas {PARAMETROS_OPCION[clave]}, se leyó '{valor}'")
        return texto

    return str(valor).strip()
