# **Introducción**

Este proyecto implementa una versión baseline del modelo BERT-Base-Uncased aplicado a la tarea de Question Answering (QA) sobre el dataset SQuAD v1.1.
El objetivo principal es entrenar y evaluar el modelo en una única GPU NVIDIA A100 del clúster FinisTerrae III (CESGA), midiendo el tiempo de ejecución total y analizando el rendimiento computacional del entrenamiento.
 

# **Contenido**

| Archivo            | Descripción                                                                                                                                               |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `job_ft3.sh`       | Script SLURM que solicita los recursos y ejecuta el entrenamiento en 1 GPU (A100).                                                                        |
| `setup_bert.sh`    | Script de instalación y configuración del entorno virtual y dependencias.                                                                                 |
| `run_qa.py`        | Script principal de entrenamiento y evaluación. Carga el modelo `bert-base-uncased`, el dataset `squad` y entrena utilizando el **Hugging Face Trainer**. |
| `trainer_qa.py`    | Definición del proceso de entrenamiento, optimizador y callbacks.                                                                                         |
| `utils_qa.py`      | Funciones auxiliares para preprocesamiento y métricas.                                                                                                    |
| `requirements.txt` | Dependencias de Python necesarias.                                                                                                                        |
| `tensorboard/`     | Logs de TensorBoard.                                                                                                                                      |
| `hf_cache/`        | Caché local de modelos y datasets de Hugging Face.                                                                                                        |
| `logs/`            | Archivos de salida y error generados por SLURM.                                                                                                           |
| `checkpoints/`     | Carpeta donde se guardan los modelos entrenados y checkpoints intermedios.                                                                                |

Debido a las restricciones de espacio para la subida de archivos por parte de github se han omitido los siguientes directorios y archivos existentes en la versión local del repositorio:
| Archivo                             | Descripción                                                                                                                              |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `hf_cache/`                         | Caché local de modelos y datasets de Hugging Face.                                                                                       |
| `checkpoints/model.safetensors`     | Modelo entrenado. No pudo subirse debido a las limitaciones de espacio de github.                                                        |
# **Instrucciones ejecucion**

1. Configurar entorno: Ejecutar el script `setup_bert.sh`, que instala los requisitos indicados en `requirements.txt` y todas las librerías necesarias para el entrenamiento.
	Ejecutar: `bash setup_bert.sh`

2. Antes de enviar el trabajo, crear carpeta logs sino existe en el directorio (debe existir antes de lanzar el sbatch).

3. Ejecutar el script `job_ft3.sh`, que solicita los recursos necesarios (GPU, CPUs, memoria, tiempo, etc.) y lanza el entrenamiento mediante el script `run_qa.py`.
	Ejecutar: `sbatch job_ft3.sh`
	El trabajo ejecuta el entrenamiento con los parámetros definidos y guarda el tiempo total de entrenamiento (`WALL_CLOCK_SECONDS`)


## **Entorno de ejecución**

- **Hardware:**

  - 1 × NVIDIA A100 (40GB)

  - 32 CPUs

  - 64 GB RAM

- **Software:**  

  - CUDA 12.2  

  - Python 3.10.8  

  - PyTorch

  - Transformers, Datasets, Accelerate, TensorBoard

- **Parámetros (BASELINE, SQuAD v1.1)**

  - timestamp: 2025-10-19 14:24:44

  - gpu_available: true

  - gpu_count: 1

  - gpu_name: NVIDIA A100-PCIE-40GB

  - cuda_version: 12.8

  - pt_version: 2.9.0+cu128
  
 
  - params:

    - model: bert-base-uncased

    - dataset: squad

    - epochs: 2

    - batch_size_per_device: 24

    - max_seq_length: 384

    - doc_stride: 128

    - lr: 4e-05

  

# **Resultados**

WALL_CLOCK_MINUTES=13, de acuerdo con este resultado el job tardó 13 minutos en completarse.

**Entrenamiento (train metrics)**

- epoch: 2.0

- train_loss: 0.9955174634695118

- train_runtime: 643.3434     # segundos ≈ 10 min 43 s

- train_samples: 88524

- train_samples_per_second: 275.2   # muestras/s

- train_steps_per_second: 11.468  # iteraciones/s


**Evaluacion (eval metrics)**

- epoch: 2.0

- eval_exact_match: 81.27719962157049     # %

- eval_f1: 88.37831928385073    # %

- eval_runtime: 21.5665 # segundos ≈ 21.57 s

- eval_samples: 10784

- eval_samples_per_second: 500.036    # muestras/s  

- eval_steps_per_second: 62.504  # iteraciones/s

**Interpretación**

- Resultados de entrenamiento: Tras dos pasadas por el dataset, la pérdida de entrenamiento final fue ~0.996. El entrenamiento (sin contar la evaluación) duró ~10 min 43 s, con una velocidad media de ~275 muestras/s y ~11.47 iteraciones/s usando la A100.

- Resultados de evaluación sobre SQuAD v1.1: el modelo alcanzó **EM ≈ 81.28%** (respuestas exactamente iguales) y **F1 ≈ 88.38%** (superposición parcial de palabras), procesando ~500 muestras/s. La evaluación completa duró ~22 s.

  Adicionalmente se incluyen los resultados de tensorboard. En concreto se seleccionar las graficas de dos métricas particularmente importantes para evaluar el aprendizaje del modelo y optimizar los recursos de entreno.

  La primera de ellas es la grafica de train/loss:
  ![alt text](https://github.com/cmr25/hpc_tools-Block2/blob/main/train_loss.PNG "train/loss")

  En ella se muestra la evolución de la función de pérdida a lo largo del entrenamiento. Como vemos a partir de los 4000 steps, aproximadamente 6 minutos desde el inicio del entrenamiento la función de pérdida se estabiliza en torno a valores bajos tal y como se esperaba esto nos indica que los pesos se ajustan adecuadamente.

  La segunda métrica es train/grad_norm:

  ![alt text](https://github.com/cmr25/hpc_tools-Block2/blob/main/train_grad_norm.PNG "train/grad_norm")
  En esta gráfica vemos la evolución de la norma del gradiente, que es una medida que nos informa de como se están ajustando los pesos del modelo para reducir la pérdida. La curva se mantiene oscilando constantemente, sin subidas descontroladas o caídas hacia cero, por lo que el entrenamiento es estable.

# **Conclusiones**

El modelo **BERT-Base-Uncased**, entrenado sobre **SQuAD v1.1**, se ejecutó correctamente en una única **GPU NVIDIA A100**, alcanzando tiempos de entrenamiento relativamente cortos (~**13 minutos** con la configuración seleccionada).
Esta implementación **baseline** servirá como punto de referencia para la versión **paralelizada** que se desarrollará en el trabajo **DISTRIBUTED**.
  

# **Referencias**

- BERT Base-uncased - HuggingFace : https://huggingface.co/google-bert/bert-base-uncased/tree/main
- Questions Answering Example: https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/README.md
- SQuAD dataset: https://rajpurkar.github.io/SQuAD-explorer/
