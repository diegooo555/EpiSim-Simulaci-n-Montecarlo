"""
Gestor de entrada/salida de archivos para secuencias de números pseudoaleatorios y semillas.
Soporta formatos .txt y .csv.
"""
import csv
import os


class ArchivoManager:
    """
    Utilidades para leer semillas y exportar secuencias generadas.
    """

    @staticmethod
    def leer_semillas(ruta_archivo):
        """
        Lee una lista de semillas enteras desde un archivo plano (.txt o .csv).

        Args:
            ruta_archivo (str): Ruta al archivo.

        Returns:
            list[int]: Lista de semillas leídas.
        """
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

        semillas = []
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    continue
                # Soporta separadores comunes: coma, punto y coma, tabulador o espacio
                partes = linea.replace(";", ",").replace("\t", ",").split(",")
                for p in partes:
                    p = p.strip()
                    if p:
                        try:
                            semillas.append(int(float(p)))
                        except ValueError:
                            pass
        return semillas

    @staticmethod
    def leer_secuencia_ri(ruta_archivo):
        """
        Lee una secuencia de números R_i (flotantes en [0, 1)) desde archivo.

        Args:
            ruta_archivo (str): Ruta al archivo .txt o .csv.

        Returns:
            list[float]: Lista de números R_i.
        """
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

        ri_list = []
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    continue
                partes = linea.replace(";", ",").replace("\t", ",").split(",")
                for p in partes:
                    p = p.strip()
                    if p:
                        try:
                            val = float(p)
                            ri_list.append(val)
                        except ValueError:
                            pass
        return ri_list

    @staticmethod
    def exportar_csv(ruta_archivo, ri_list, xi_list=None, ni_list=None):
        """
        Exporta los datos generados a un archivo CSV.

        Args:
            ruta_archivo (str): Ruta de destino.
            ri_list (list[float]): Números uniformes R_i.
            xi_list (list[int], opcional): Semillas/enteros generados X_i.
            ni_list (list[float], opcional): Números transformados N_i.
        """
        n = len(ri_list)
        columnas = ["i", "Ri"]
        if xi_list is not None:
            columnas.insert(1, "Xi")
        if ni_list is not None:
            columnas.append("Ni")

        with open(ruta_archivo, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columnas)
            for i in range(n):
                fila = [i + 1]
                if xi_list is not None:
                    # xi_list suele tener x0 al inicio
                    val_xi = xi_list[i + 1] if len(xi_list) > n else xi_list[i]
                    fila.append(val_xi)
                fila.append(f"{ri_list[i]:.6f}")
                if ni_list is not None and i < len(ni_list):
                    fila.append(f"{ni_list[i]:.6f}")
                writer.writerow(fila)

    @staticmethod
    def exportar_txt(ruta_archivo, ri_list):
        """
        Exporta una secuencia de números R_i a un archivo de texto simple, un número por línea.

        Args:
            ruta_archivo (str): Ruta de destino.
            ri_list (list[float]): Números R_i.
        """
        with open(ruta_archivo, "w", encoding="utf-8") as f:
            for r in ri_list:
                f.write(f"{r:.6f}\n")
