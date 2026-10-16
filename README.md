# Proyecto CNN BloodMNIST

Proyecto de curso para estudiar, mediante experimentos controlados, cómo BatchNorm, Dropout y la tasa de aprendizaje afectan una CNN que clasifica imágenes de células sanguíneas. El flujo sigue la guía: explorar los datos, formular una pregunta e hipótesis, comparar con evidencia y reservar test para la evaluación final.

## Contexto de la investigación

Las redes neuronales convolucionales (CNN) permiten aprender patrones visuales a partir de imágenes y se utilizan en tareas de clasificación. Sin embargo, el resultado de un modelo no depende únicamente de su arquitectura: también puede verse influido por las técnicas de normalización y regularización, así como por los parámetros empleados durante el entrenamiento.

Este proyecto utiliza BloodMNIST como caso de estudio para examinar, mediante experimentos controlados, cómo determinadas decisiones de diseño y entrenamiento se relacionan con el aprendizaje y la clasificación de imágenes de células sanguíneas. El interés no consiste solamente en obtener una métrica elevada, sino en observar diferencias, analizar posibles errores y formular conclusiones respaldadas por evidencia.

## Justificación

Comparar configuraciones bajo condiciones definidas permite estudiar el efecto de cambios concretos sin atribuir automáticamente cualquier diferencia a una sola causa. La CNN base funciona como referencia para interpretar los resultados de las variantes experimentales. Mantener constantes los parámetros que no son objeto de estudio contribuye a que la comparación sea más clara y reproducible.

El proyecto también sigue una lógica de investigación basada en evidencia: primero se exploran los datos; después se formula una pregunta e hipótesis; luego se ejecutan y comparan los experimentos; finalmente, se interpretan las métricas, las curvas de aprendizaje y los errores observados. Los resultados pueden respaldar una hipótesis, apoyarla parcialmente o no respaldarla. En todos los casos, las conclusiones deben derivarse de los resultados reales y reconocer las limitaciones del diseño.

## Pregunta de investigación

**¿Cómo influyen las configuraciones de la CNN evaluadas en este proyecto en el desempeño de clasificación de BloodMNIST y en la evolución de las pérdidas de entrenamiento y validación?**

La pregunta se concreta en los factores que distinguen los escenarios activos: la incorporación de BatchNorm, el uso de Dropout y el cambio de la tasa de aprendizaje. La comparación se realiza respecto de la CNN base y de acuerdo con la configuración documentada en la sección de escenarios.

## Hipótesis de investigación

La hipótesis general plantea que **los cambios de normalización, regularización y tasa de aprendizaje pueden producir diferencias observables en la dinámica de entrenamiento y en el desempeño de validación de la CNN**, aunque no necesariamente todas las modificaciones mejorarán todas las métricas.

Como expectativas específicas por contrastar:

- **BatchNorm:** podría contribuir a una evolución más estable del entrenamiento o a un mejor desempeño de validación.
- **Dropout:** podría ayudar a reducir el sobreajuste, aunque también podría disminuir el desempeño de entrenamiento o no generar una mejora medible.
- **Tasa de aprendizaje menor:** podría generar una evolución más gradual de la pérdida, pero necesitar más épocas para alcanzar un desempeño comparable.

Estas afirmaciones son hipótesis y no resultados anticipados. Solo se aceptarán, matizarán o rechazarán a partir de la evidencia obtenida en los runs oficiales.

## Objetivos

### Objetivo general

Evaluar mediante experimentos controlados cómo las configuraciones activas de la CNN influyen en el entrenamiento y en la clasificación de imágenes de BloodMNIST.

### Objetivos específicos

- Explorar las características y la distribución de clases del conjunto de datos antes del entrenamiento.
- Comparar las configuraciones activas manteniendo constantes las condiciones que no corresponden al factor estudiado.
- Examinar las métricas y la evolución de las pérdidas de entrenamiento y validación.
- Revisar los errores de clasificación y las diferencias entre clases cuando los artefactos disponibles lo permitan.

## Metodología de investigación

El proyecto adopta un enfoque experimental cuantitativo. Se parte de una exploración descriptiva del conjunto de datos y se continúa con la comparación de configuraciones de una CNN. Cada escenario representa una condición experimental definida en la configuración del proyecto.

El procedimiento general es:

1. **Exploración de datos (EDA):** revisar la estructura del conjunto de datos, la distribución de clases y ejemplos de imágenes para contextualizar el problema.
2. **Definición de referencia:** utilizar la CNN base como punto de comparación.
3. **Formulación de hipótesis:** establecer expectativas razonables sobre el efecto de los cambios experimentales, sin asumir que producirán mejoras.
4. **Ejecución de los runs:** entrenar los escenarios activos bajo los parámetros compartidos definidos en el apartado técnico, modificando únicamente los factores que diferencian cada escenario.
5. **Seguimiento y selección:** utilizar el conjunto de validación para observar el aprendizaje y seleccionar el checkpoint de acuerdo con el criterio técnico configurado.
6. **Evaluación final:** reservar el conjunto de prueba para la evaluación final, después de seleccionar el modelo mediante validación.
7. **Análisis e interpretación:** comparar métricas y curvas, revisar errores disponibles y determinar qué evidencia respalda o contradice las hipótesis.
8. **Conclusiones y limitaciones:** responder la pregunta de investigación, describir los hallazgos comprobados y reconocer los límites del experimento.

### Variables y control experimental

Los factores de interés son las configuraciones de normalización y regularización, y la tasa de aprendizaje según los escenarios activos. Las métricas de entrenamiento y validación constituyen evidencia para comparar su comportamiento y desempeño. Los demás parámetros se mantienen según la configuración técnica del proyecto, con el fin de facilitar una comparación controlada.

El uso de una semilla fija favorece la reproducibilidad de las ejecuciones, pero no permite medir por sí solo la variabilidad entre distintas semillas. Por ello, los resultados deben interpretarse dentro del alcance de los runs realizados y no como una demostración universal de que una configuración será superior en todos los contextos.

### Criterios para interpretar los resultados

La interpretación no debe depender exclusivamente de la exactitud (*accuracy*). Se considerarán las métricas disponibles en el proyecto, la evolución de las pérdidas de entrenamiento y validación y, cuando existan, las matrices de confusión y los resultados por clase. También se examinarán posibles diferencias entre entrenamiento y validación como indicios de sobreajuste, sin atribuir causalidad más allá de lo que permitan los experimentos.

Los resultados y las conclusiones definitivas se incorporarán una vez ejecutados y verificados los runs oficiales. No deben presentarse resultados de pruebas piloto como si fueran resultados definitivos ni completar esta sección con conclusiones que no estén sustentadas por los artefactos generados.

## Escenarios

| ID | Configuración | Cambio frente a la CNN base |
|---|---|---|
| A1 | CNN base | Referencia |
| A2 | CNN + BatchNorm | BatchNorm |
| A3 | CNN + BatchNorm + Dropout | BatchNorm y Dropout (p=0.3) |
| D1 | CNN + Dropout | Dropout (p=0.3), sin BatchNorm |
| C1 | CNN base | Tasa de aprendizaje 0.0003 |

Las runs usan semilla 42, CPU, batch size 128 y hasta 40 épocas. Se conserva el checkpoint con menor pérdida de validación. B1 (early stopping) se retiró del plan activo; sus archivos permanecen como registro histórico y no entran en la comparación vigente.

## Estructura

- `configs/experiments.json`: parámetros y escenarios activos.
- `src/bloodmnist/`: carga de datos, EDA, modelo, entrenamiento y métricas.
- `scripts/`: comandos de EDA, entrenamiento y evaluación final.
- `notebooks/01_orquestacion.ipynb`: ejecuta el flujo y presenta los artefactos guardados; la lógica principal está en módulos y scripts.
- `outputs/`: informes, historiales y figuras existentes. Los checkpoints `.pt` no se incluyen en Git.

## Preparar el entorno en Windows

Clona o descarga el repositorio y abre PowerShell en la carpeta `Proyecto_CNN_BloodMNIST`. Crea el entorno e instala primero PyTorch para CPU y luego las dependencias restantes:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación, permite scripts solo en la sesión actual y vuelve a activar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Ejecutar el proyecto

Abre el notebook orquestador desde la raíz del repositorio:

```powershell
jupyter lab notebooks/01_orquestacion.ipynb
```

Ejecuta las celdas en orden. El EDA descarga BloodMNIST la primera vez y genera sus resúmenes y figuras. Las secciones de informe leen los archivos de `outputs/`; no vuelven a entrenar si se mantienen comentadas las celdas opcionales de entrenamiento.

Para regenerar el EDA o entrenar desde PowerShell, usa estos comandos desde la raíz:

```powershell
python scripts/run_eda.py
python scripts/run_training.py --scenario A1
```

Para correr todas las configuraciones activas en secuencia —hasta 40 épocas cada una—:

```powershell
python scripts/run_training.py --all
```

Una nueva corrida reemplaza los historiales y checkpoints de los escenarios ejecutados. La evaluación final de test es independiente y se hace una sola vez, después de escoger el checkpoint usando validation. El resultado existente corresponde a A3; no vuelvas a evaluar test si `outputs/final_evaluation/test_metrics.json` ya existe.

## Artefactos incluidos

Los informes, historiales, resúmenes y figuras de `outputs/` se conservan para que se pueda revisar el proyecto sin repetir el entrenamiento. Los pesos `.pt` se excluyen; si se vuelve a ejecutar el informe tras clonar, primero hay que entrenar los escenarios para crear esos checkpoints. Los datos descargados también se excluyen y MedMNIST los obtiene al ejecutar el EDA o el entrenamiento.
