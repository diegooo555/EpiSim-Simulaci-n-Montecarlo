# EpiSim — Simulación Montecarlo de propagación de enfermedades (Punto 4)

Simulación estocástica basada en individuos de un modelo **SEIR** (Susceptible → Expuesto → Infeccioso → Recuperado/Removido) para una población de 10.000 personas durante 365 días, repetida 1.000 veces (Montecarlo). Todos los números aleatorios provienen del módulo `pseudo_gen` (desarrollado desde cero en el punto 3); no se usa `random` ni `numpy.random`.

---

## Estructura

```text
4_sim_enf/
├── episim.py          # Programa principal (línea de comandos)
├── configuracion.py   # Lectura y validación de parámetros desde .csv / .txt
├── aleatorio.py       # Integración con pseudo_gen: flujos independientes, transformaciones, validación
├── modelo.py          # Modelo SEIR: contactos, selección, transmisión, progresión de estados
├── montecarlo.py      # Ejecución paralela de N simulaciones con validación de los R_i usados
├── analisis.py        # Estadísticas descriptivas, pruebas de diferencia, sensibilidad, cuadro del equipo
├── graficas.py        # Gráficas (Matplotlib)
├── config/
│   ├── con_vacunacion.csv     # Escenario con vacunación (parámetros del enunciado)
│   ├── sin_vacunacion.csv     # Igual, con cobertura de vacunación = 0
│   └── ejemplo_formato.txt    # Ejemplo del formato .txt (lectura literal del enunciado)
├── pseudo_gen/        # Módulo de generadores y pruebas (punto 3)
└── resultados/        # Salidas (se crean al ejecutar)
```

## Instalación y uso

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

python episim.py todo            # Ejecuta todo: equipo, flujos, traza, comparación y sensibilidad
python episim.py interfaz        # Abre la interfaz gráfica
```

También podés abrirla directamente con `python gui.py`. La interfaz requiere Tkinter
(incluido normalmente en Python; en algunas distribuciones Linux se instala por separado,
por ejemplo con el paquete `python3-tk`) y un entorno gráfico disponible.

| Comando | Qué hace |
| :--- | :--- |
| `python episim.py escenario config/con_vacunacion.csv` | Montecarlo de un escenario: tablas y gráficas |
| `python episim.py comparar config/sin_vacunacion.csv config/con_vacunacion.csv` | Ejecuta ambos escenarios y los compara |
| `python episim.py sensibilidad config/con_vacunacion.csv --n 200` | Sensibilidad un parámetro a la vez (tornado) |
| `python episim.py traza --flujo 0 --dias 2` | Una simulación con el detalle del uso de cada R_i |
| `python episim.py flujos` | Verifica que las simulaciones no estén correlacionadas entre sí |
| `python episim.py equipo` | Cuadro técnico (lenguaje, SO, procesador, RAM) |
| `python episim.py interfaz` | Abre la interfaz gráfica de EpiSim |

Opciones comunes: `--n` (simulaciones; por defecto el valor del archivo), `--procesos` (por defecto todos los núcleos), `--salida` (por defecto `resultados`).

---

## Parametrización externa

Los escenarios se definen en archivos `.csv` (`parametro,valor,descripcion`) o `.txt` (`parametro = valor`). Un parámetro estocástico se escribe `U(a, b)`; un valor fijo `v` equivale a `U(v, v)`. Lo que no aparezca en el archivo toma el valor por defecto (el del enunciado).

| Parámetro | Por defecto | Se muestrea | Descripción |
| :--- | :--- | :--- | :--- |
| `poblacion` | 10000 | — | N |
| `infectados_iniciales` | 10 | — | Casos índice (en I el día 0) |
| `dias` | 365 | — | Horizonte |
| `n_simulaciones` | 1000 | — | Simulaciones aceptadas |
| `beta` | U(0.3, 0.5) | por simulación | Contactos promedio por día de cada infeccioso |
| `periodo_incubacion` | U(2, 5) | por individuo | Días en E (entero uniforme) |
| `periodo_infeccioso` | U(7, 14) | por individuo | Días en I (entero uniforme) |
| `p_transmision` | U(0.4, 0.6) | por simulación | Prob. de contagio si hay contacto |
| `cobertura_vacunacion` | U(0.3, 0.7) | por simulación | Proporción vacunada |
| `efectividad_vacuna` | U(0.80, 0.95) | por simulación | Reducción de susceptibilidad |
| `letalidad` | U(0.005, 0.015) | por simulación | **Supuesto propio** (el enunciado no la da): prob. de fallecer al salir de I |
| `modo_contactos` | `estocastico` | — | `estocastico`: piso(R + β) · `literal`: piso(R · β) |
| `seleccion_contacto` | `poblacion` | — | `poblacion`: contacto entre los N · `susceptibles`: entre los S |
| `umbral_control` | 0 | — | Brote controlado cuando E + I ≤ umbral (después del pico) |
| `generador` | `lcg` | — | `lcg` (Congruencial Lineal) o `mult` (Congruencial Multiplicativo) |
| `semilla_maestra` | 12345 | — | Semilla de la que se derivan los flujos |
| `archivo_semillas` | (vacío) | — | Opcional: archivo con semillas explícitas (una por simulación) |
| `validar_numeros` | `si` | — | Aplicar las 5 pruebas a los R_i usados por cada simulación |
| `alpha` | 0.05 | — | Nivel de significancia |

---

## Mecanismo estocástico

### Generador y documentación del método usado

| Uso | Generador (`pseudo_gen`) | Transformación |
| :--- | :--- | :--- |
| β, p, cobertura, efectividad, letalidad | Congruencial Lineal, a = 1664525, c = 1013904223, m = 2³² | `pseudo_gen.uniforme`: a + (b − a)·R |
| Períodos de incubación e infeccioso | El mismo flujo | Uniforme discreta: a + piso(R·(b − a + 1)) |
| Contactos, selección, transmisión, vacunación, letalidad | El mismo flujo | Uso directo de R ~ U(0,1) |

El generador conserva su estado, así que cada simulación consume una única secuencia R₁, R₂, … en el orden descrito a continuación.

### Orden de consumo de los R_i en una simulación

1. **Parámetros** (5 R_i): β, p, cobertura, efectividad, letalidad ~ U(a, b).
2. **Casos índice**: individuo = piso(R·N) (se repite si ya está infectado) + 1 R_i para su período infeccioso.
3. **Vacunación**: un R_i por individuo; si R < cobertura queda vacunado con factor de susceptibilidad 1 − efectividad. Se consume el R_i aunque la cobertura sea 0 para que los escenarios con y sin vacunación usen los mismos números en el resto de la simulación (*números aleatorios comunes*), lo que hace más precisa la comparación.
4. **Cada día** t = 1…365:
   - **Progresión** (calendario de eventos): I → R de quienes cumplen su período (1 R_i: R < letalidad → fallece); E → I de quienes cumplen la incubación (1 R_i: período infeccioso).
   - **Paso 1 – contactos**: para cada infeccioso, R → número de contactos.
   - **Paso 2 – selección**: R → individuo piso(R·N).
   - **Paso 3 – transmisión**: R < p·factor_vacunación → el contactado pasa a E (1 R_i: período de incubación).

`python episim.py traza` imprime y guarda (`resultados/<escenario>/traza_flujo_0.txt`) cada R_i con la operación que produjo.

### Decisiones de modelado (y por qué)

- **Contactos = piso(R + β), no piso(R · β).** Con β ∈ [0.3, 0.5] la fórmula literal piso(R·β) siempre da 0 (el propio ejemplo del enunciado da 0 contactos), así que la enfermedad nunca se transmitiría. piso(R + β) conserva la forma "piso de una expresión con R", toma los valores piso(β) o piso(β) + 1 y su media es exactamente β, que es como el enunciado define β ("promedio de contactos por día"). La fórmula literal sigue disponible (`modo_contactos = literal`, ver `config/ejemplo_formato.txt`) para mostrar el problema: con ella solo se infectan los 10 casos índice.
- **Contacto elegido entre toda la población.** Si el contactado no es susceptible, el contacto no contagia. Así la probabilidad de encontrar a un susceptible es S/N (mezcla homogénea, como en el SEIR clásico) y la epidemia se frena al agotarse los susceptibles. La lectura literal (elegir siempre a un susceptible) está en `seleccion_contacto = susceptibles`; con ella la fuerza de infección no disminuye con S y, sin vacunación, se infecta el 100 % de la población en todas las simulaciones (verificado con 20 corridas).
- **Períodos por individuo, enteros.** Cada persona recibe su propia incubación e infecciosidad al entrar al estado; después la progresión es determinística, como pide el enunciado.
- **Número reproductivo básico implícito**: R₀ ≈ β · p · D ≈ 0.4 · 0.5 · 10.5 ≈ 2.1 (entre ≈ 1.3 y ≈ 3.2 según los β y p muestreados, con D = 10.5 días en promedio).

### Independencia entre simulaciones (flujos)

La simulación k usa la semilla *x*₀ = estado del generador tras k·SALTO pasos desde la semilla maestra, calculado con la fórmula de salto adelante del congruencial en O(log k):

x_{n+k} = aᵏ·x_n + c·(aᵏ − 1)/(a − 1) (mod m)

Así cada simulación usa un tramo del ciclo que no se solapa con los demás (2.097.153 números por flujo; una simulación usa ~10⁴–2·10⁵).

Se eligió **SALTO = 2²¹ + 1** y no 2²¹: con m = 2³², a^(2²¹) ≡ 1 (mod 2²³), por lo que el primer número de cada flujo sería casi el del flujo anterior más una constante (correlación 0.52; ver `resultados/flujos/dispersion_flujos.png`). Con el salto impar la secuencia "primer R_i de cada flujo" aprueba las 5 pruebas (`python episim.py flujos`).

### Validación de los números usados

`FlujoAleatorio` registra cada R_i que consume la simulación. Al terminar, esa secuencia exacta se somete a las 5 pruebas de `pseudo_gen` (medias, varianza, χ², KS, póker) con α = 0.05. Si falla alguna, la simulación se **rechaza** y se usa el flujo siguiente, hasta completar las 1.000 aceptadas. Como hay 5 pruebas a α = 0.05, se espera rechazar ≈ 1 − 0.95⁵ ≈ 23 % aun con un generador perfecto; `validacion_numeros.png` compara la tasa de aprobación de cada prueba con el 95 % esperado.

---

## Salidas

Por escenario (`resultados/<nombre>/`):

| Archivo | Contenido |
| :--- | :--- |
| `simulaciones.csv` | Una fila por simulación aceptada: flujo, semilla, parámetros muestreados, métricas, R_i usados, tiempos |
| `rechazadas.csv` | Simulaciones rechazadas y la prueba que fallaron |
| `series.npz` | Series S, E, I, R diarias de todas las simulaciones (arreglo n × 366 × 4) |
| `estadisticas.csv` | Media, desviación, IC 95 %, percentiles de cada métrica |
| `curva_epidemica.png` | S, E, I, R promedio con IC 95 % del promedio y banda P2.5–P97.5 |
| `hist_*.png` | Pico (magnitud y día), total de infectados, fallecidos, días hasta control |
| `parametros_muestreados.png` | Histogramas de los parámetros U(a, b) muestreados |
| `validacion_numeros.png` | % de simulaciones que aprueban cada prueba |
| `sensibilidad_spearman.png` | Correlación parámetro–resultado (sensibilidad global) |

Además: `comparacion/` (curvas y cajas sin vs. con vacunación, prueba de Mann-Whitney), `sensibilidad/` (tornado y tabla), `flujos/` y `equipo.csv`.

### Métricas

- **Total de infectados**: casos índice + todos los que pasaron a E. **Tasa de ataque** = total / N.
- **Pico**: máximo de I(t) y el día en que ocurre.
- **Día de control**: primer día posterior al pico con E + I ≤ `umbral_control` (0 = extinción del brote). Si no ocurre en 365 días, se cuenta como "no controlado" y se excluye del histograma (se indica cuántos).
- **Banda de confianza**: se grafican dos bandas: el IC 95 % del promedio (qué tan preciso es el promedio Montecarlo) y el intervalo P2.5–P97.5 (qué tan variable es una epidemia individual).

### Sensibilidad

- **Un parámetro a la vez (tornado)**: cada parámetro se fija en su mínimo y en su máximo mientras los demás siguen muestreándose; 200 simulaciones por caso, con los mismos flujos en todos los casos (números aleatorios comunes).
- **Global (Spearman)**: correlación de rangos entre los parámetros muestreados y los resultados de las 1.000 simulaciones del escenario, sin corridas adicionales.

---

## Limitaciones conocidas

- Mezcla homogénea: todos pueden contactar a todos con igual probabilidad (sin hogares, edades ni geografía).
- Parámetros constantes en el tiempo: no hay cambios de comportamiento, cuarentenas ni vacunación durante el brote.
- La vacuna solo reduce la susceptibilidad; no reduce la transmisión de un vacunado infectado ni su letalidad.
- La letalidad es un supuesto (el enunciado no la especifica) y no depende de la edad ni de la capacidad hospitalaria.
- Máximo un contacto por infeccioso al día con los rangos del enunciado (β < 1).
- Los bits bajos de un LCG con m = 2³² tienen períodos cortos; aquí solo se usan R = x/m (dominado por los bits altos) y las secuencias se validan.
- La prueba de varianza de `pseudo_gen` usa el intervalo χ² del curso, que supone datos normales. Para R_i uniformes la varianza muestral fluctúa menos (curtosis 1.8 frente a 3), así que el intervalo es más ancho de lo necesario y la prueba casi nunca rechaza (100 % de aprobación en `validacion_numeros.png`, frente al 95 % nominal). Las otras cinco pruebas se comportan como se espera.
