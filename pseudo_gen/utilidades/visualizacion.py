"""
Módulo de visualización gráfica para generadores y pruebas estadísticas.
Genera los gráficos exigidos en el taller de simulación usando Matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np


class Visualizacion:
    """
    Colección de métodos estáticos para generar los gráficos requeridos por el taller.
    """

    @staticmethod
    def _finalizar_figura(fig, guardar_en=None, mostrar=True):
        plt.tight_layout()
        if guardar_en:
            fig.savefig(guardar_en, dpi=300)
        if mostrar:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def histograma_generacion(datos, titulo="Histograma de Números Generados", bins=20, guardar_en=None, mostrar=True):
        """
        Histograma de frecuencias para visualizar la distribución alcanzada.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(datos, bins=bins, color="#4A90E2", edgecolor="#1C3F75", alpha=0.75, density=True)
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xlabel("Valor", fontsize=10)
        ax.set_ylabel("Densidad de Frecuencia", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        Visualizacion._finalizar_figura(fig, guardar_en, mostrar)
        return fig

    @staticmethod
    def prueba_medias(resultado, guardar_en=None, mostrar=True):
        """
        Gráfico para la Prueba de Medias:
        Muestra la media calculada, límites de confianza y media teórica (0.5).
        """
        det = resultado.detalles
        media_calc = det["media_calculada"]
        lim_inf = det["limite_inferior"]
        lim_sup = det["limite_superior"]
        media_teo = det["media_esperada"]

        fig, ax = plt.subplots(figsize=(8, 5))
        categorias = ["Límite Inferior", "Media Calculada", "Media Teórica", "Límite Superior"]
        valores = [lim_inf, media_calc, media_teo, lim_sup]
        colores = ["#E74C3C", "#2ECC71" if resultado.aprobada else "#E67E22", "#3498DB", "#E74C3C"]

        barras = ax.bar(categorias, valores, color=colores, width=0.5, edgecolor="black")
        ax.axhline(media_teo, color="#2980B9", linestyle="--", alpha=0.7, label=f"Teórica ({media_teo})")
        ax.set_title(f"Prueba de Medias - {'APROBADA' if resultado.aprobada else 'RECHAZADA'}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Valor Promedio", fontsize=10)

        for bar, val in zip(barras, valores):
            ax.annotate(f"{val:.5f}",
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

        ax.legend()
        Visualizacion._finalizar_figura(fig, guardar_en, mostrar)
        return fig

    @staticmethod
    def prueba_varianza(resultado, guardar_en=None, mostrar=True):
        """
        Gráfico para la Prueba de Varianza:
        Muestra varianza calculada, límites de confianza y varianza teórica (1/12).
        """
        det = resultado.detalles
        var_calc = det["varianza_calculada"]
        lim_inf = det["limite_inferior"]
        lim_sup = det["limite_superior"]
        var_teo = det["varianza_esperada"]

        fig, ax = plt.subplots(figsize=(8, 5))
        categorias = ["Límite Inferior", "Varianza Calculada", "Varianza Teórica", "Límite Superior"]
        valores = [lim_inf, var_calc, var_teo, lim_sup]
        colores = ["#E74C3C", "#2ECC71" if resultado.aprobada else "#E67E22", "#3498DB", "#E74C3C"]

        barras = ax.bar(categorias, valores, color=colores, width=0.5, edgecolor="black")
        ax.axhline(var_teo, color="#2980B9", linestyle="--", alpha=0.7, label=f"Teórica ({var_teo:.5f})")
        ax.set_title(f"Prueba de Varianza - {'APROBADA' if resultado.aprobada else 'RECHAZADA'}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Varianza", fontsize=10)

        for bar, val in zip(barras, valores):
            ax.annotate(f"{val:.5f}",
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

        ax.legend()
        Visualizacion._finalizar_figura(fig, guardar_en, mostrar)
        return fig

    @staticmethod
    def prueba_chi2(resultado, guardar_en=None, mostrar=True):
        """
        Gráfico para la Prueba Chi-Cuadrado:
        Barras comparando frecuencias observadas vs esperadas por intervalo.
        """
        det = resultado.detalles
        k = det["k_intervalos"]
        limites = det["limites_intervalos"]
        obs = det["frecuencias_observadas"]
        esp = det["frecuencia_esperada"]

        etiquetas = [f"[{limites[i]:.2f}, {limites[i+1]:.2f})" for i in range(k)]
        x = np.arange(k)
        ancho = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - ancho/2, obs, ancho, label="Observada (Oi)", color="#3498DB", edgecolor="black")
        ax.bar(x + ancho/2, [esp] * k, ancho, label="Esperada (Ei)", color="#E67E22", alpha=0.8, edgecolor="black")

        ax.set_title(f"Prueba Chi-Cuadrado de Uniformidad - {'APROBADA' if resultado.aprobada else 'RECHAZADA'}\n"
                     f"χ₀² = {resultado.estadistico:.4f} | χ_crit² = {resultado.valor_critico:.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Intervalos", fontsize=10)
        ax.set_ylabel("Frecuencia", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, rotation=35, ha="right", fontsize=9)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        Visualizacion._finalizar_figura(fig, guardar_en, mostrar)
        return fig

    @staticmethod
    def prueba_ks(resultado, guardar_en=None, mostrar=True):
        """
        Gráfico para la Prueba Kolmogorov-Smirnov (KS):
        Función Empírica Acumulada (FEC) vs Teórica Uniforme F(x)=x, destacando D_max.
        """
        det = resultado.detalles
        ordenados = det["numeros_ordenados"]
        fec_y = det["fec_y"]
        d_max = det["d_max"]
        d_critico = det["d_critico"]
        punto_max = det["punto_d_max"]

        fig, ax = plt.subplots(figsize=(8, 6))
        # Curva empírica paso a paso
        ax.step(ordenados, fec_y, label="Empírica Acumulada Sn(x)", color="#2980B9", where="post", linewidth=2)
        # Curva teórica
        ax.plot([0, 1], [0, 1], label="Teórica F(x) = x", color="#E74C3C", linestyle="--", linewidth=2)

        # Destacar punto de máxima discrepancia
        ax.plot([punto_max[0], punto_max[0]], [punto_max[0], punto_max[1]],
                color="#8E44AD", linestyle=":", linewidth=2.5,
                label=f"D_max = {d_max:.4f} (Crítico: {d_critico:.4f})")
        ax.scatter([punto_max[0]], [punto_max[1]], color="#8E44AD", s=60, zorder=5)

        ax.set_title(f"Prueba Kolmogorov-Smirnov - {'APROBADA' if resultado.aprobada else 'RECHAZADA'}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("x (Valor)", fontsize=10)
        ax.set_ylabel("Probabilidad Acumulada", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")

        Visualizacion._finalizar_figura(fig, guardar_en, mostrar)
        return fig

    @staticmethod
    def prueba_poker(resultado, guardar_en=None, mostrar=True):
        """
        Gráfico para la Prueba de Póker:
        Comparación de frecuencias observadas vs esperadas por cada una de las 7 manos.
        """
        det = resultado.detalles
        categorias = det["categorias"]
        obs = det["frecuencias_observadas"]
        esp = det["frecuencias_esperadas"]

        # Abreviaciones para el eje X
        nombres_cortos = ["D (Dif)", "O (1 Par)", "T (2 Pares)", "K (Tercia)",
                          "F (Full)", "P (Póker)", "Q (Quintilla)"]
        x = np.arange(len(nombres_cortos))
        ancho = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - ancho/2, obs, ancho, label="Observadas (Oi)", color="#27AE60", edgecolor="black")
        ax.bar(x + ancho/2, esp, ancho, label="Esperadas (Ei)", color="#F39C12", alpha=0.8, edgecolor="black")

        ax.set_title(f"Prueba de Póker (5 dígitos) - {'APROBADA' if resultado.aprobada else 'RECHAZADA'}\n"
                     f"χ₀² = {resultado.estadistico:.4f} | χ_crit² = {resultado.valor_critico:.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Manos de Póker", fontsize=10)
        ax.set_ylabel("Frecuencia", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(nombres_cortos, fontsize=9)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        Visualizacion._finalizar_figura(fig, guardar_en, mostrar)
        return fig
