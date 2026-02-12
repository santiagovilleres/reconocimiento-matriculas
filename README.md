# Sistema de Reconocimiento Automático de Matrículas Vehiculares Argentinas Mediante Visión por Computadora y OCR en Python

Instituto Superior De Formación Técnica N°130.
Tecnicatura Superior En Análisis De Sistemas.
Algoritmos y Estructuras De Datos III.


## 1. Resumen

El presente proyecto desarrolla un sistema de Reconocimiento Automático de Matrículas mediante técnicas de Visión por Computador.

El sistema implementa un pipeline completo que abarca:

* Detección de matrículas mediante un modelo YOLOv8 entrenado.
* Recorte y preprocesamiento de la región detectada.
* Reconocimiento de caracteres utilizando Fast Plate OCR.
* Postprocesado de texto.
* Evaluación cuantitativa mediante métricas de detección y reconocimiento.

Se garantiza reproducibilidad mediante scripts organizados y entorno controlado.


## 2. Objetivo

Diseñar e implementar un sistema capaz de:

1. Detectar automáticamente la región de la matrícula en imágenes o webcam.
2. Extraer el texto completo de la placa.
3. Evaluar el desempeño utilizando métricas detección y OCR.
4. Presentar resultados cuantificables y analizables.


## 3. Arquitectura del Sistema

El flujo lógico implementado es el siguiente:

```
Imagen / Webcam
      ↓
Detección (YOLOv8)
      ↓
Recorte de región
      ↓
Preprocesamiento
      ↓
OCR (Fast Plate OCR)
      ↓
Postprocesamiento
      ↓
Texto final
```


## 4. Tecnologías Utilizadas

* Python 3.10+
* Ultralytics YOLOv8
* Fast Plate OCR (ONNX Runtime)
* OpenCV
* NumPy
* Pandas


## 5. Estructura del Proyecto

```
project/
├── data/
│    ├──  train/
│    ├── val/
│    └── test/
├── models/
│   └── weights/
│       └── best.pt
├── src/
│   ├── train.py
│   ├── detect.py
│   ├── ocr.py
│   ├── evaluate.py
│   └── dataset.py
├── requirements.txt
└── README.md
```


## 6. Instalación y Reproducibilidad

Clonar repositorio:

```bash
git clone https://github.com/santiagovilleres/reconocimiento-matriculas
cd reconocimiento-matriculas
```

Crear entorno virtual (Python 3.10/11 recomendado):

```bash
python -m venv venv
```

Activar entorno:

```bash
venv\Scripts\activate
```

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```


## 7. Ejecución

### Inferencia sobre imagen

```bash
python src/detect.py --image ruta_imagen.jpg
```

### Evaluación sobre dataset de test

```bash
python src/evaluate.py
```

## 8. Métricas Implementadas

### Detección

* Precision
* Recall
* F1-score
* mAP@0.5
* IoU promedio

### Reconocimiento OCR

* CER (Character Error Rate)
* Accuracy por matrícula completa

### Rendimiento

* Latencia promedio por imagen
* Comparación CPU/GPU (si aplica)

## 9. Autores

* Tomás S. Leira, Simon Ortiz, Santiago Villeres
* Instituto Superior De Formación Técnica N°130
* Tecnicatura Superior En Análisis De Sistemas
* Algoritmos y Estructuras De Datos III
* Docente: Pablo Letier
* 23 de febrero de 2026