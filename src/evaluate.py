from ultralytics import YOLO
from ocr import OCR
import cv2
import os
import time
import psutil
import torch
import numpy as np
from Levenshtein import distance as distancia_levenshtein

RUTA_MODELO = "models/weights/best.pt"
NOMBRE_MODELO_OCR = "cct-xs-v1-global-model"

RUTA_DATOS = "data/data.yaml"
RUTA_IMAGENES_TEST = "data/test/images"

RUTA_LABELS_TEST = "data/test/labels"

RUTA_IMAGENES_OCR = "data/ocr/images"
RUTA_LABELS_OCR = "data/ocr/labels.txt"

class Evaluador:
    def __init__(self):
        self.modelo = YOLO(RUTA_MODELO)
        self.modelo_ocr = OCR(NOMBRE_MODELO_OCR)

    def evaluar(self):

        precision, recall, f1, mapa50 = self._evaluar_deteccion()
        print(f"mAP@0.5: {mapa50:.4f}")
        print(f"Precisión: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        
        iou_media = self._calcular_iou_media()
        print(f"IoU media: {iou_media:.4f}")

        error_caracter, precision_ocr, contador = self._evaluar_ocr()
        print(f"OCR - imágenes procesadas: {contador}")
        print(f"CER: {error_caracter:.4f}")
        print(f"Precisión: {precision_ocr:.4f}")

        latencia = self._medir_latencia_lote()
        print(f"Latencia promedio por imagen: {latencia:.2f} ms")

        latencia_webcam = self._medir_latencia_webcam()

        if latencia_webcam:
            print(f"Latencia promedio webcam: {latencia_webcam:.2f} ms/frame")
        else:
            print("No se pudo medir latencia webcam")

        ram, gpu = self._medir_recursos()
        print(f"RAM usada: {ram:.2f} MB")
        if gpu:
            print(f"Memoria GPU usada: {gpu:.2f} MB")
        else:
            print("GPU no disponible")

    def _evaluar_deteccion(self):
        resultados = self.modelo.val(data=RUTA_DATOS, split='test', verbose=False)
        precision = resultados.box.mp
        recall = resultados.box.mr
        f1 = 2 * precision * recall / (precision + recall)
        mapa50 = resultados.box.map50

        return precision, recall, f1, mapa50

    def _calcular_iou_media(self):        
        ious = []

        for imagen in sorted(os.listdir(RUTA_IMAGENES_TEST)):

            ruta_imagen = RUTA_IMAGENES_TEST + "/" + imagen
            ruta_label = RUTA_LABELS_TEST + "/" + imagen.rsplit('.', 1)[0] + '.txt'

            imagen = cv2.imread(ruta_imagen)

            altura, ancho = imagen.shape[:2]

            with open(ruta_label, 'r') as archivo:
                ground_truth = [[float(p[1])*ancho - float(p[3])*ancho/2, float(p[2])*altura - float(p[4])*altura/2,
                       float(p[1])*ancho + float(p[3])*ancho/2, float(p[2])*altura + float(p[4])*altura/2]
                      for p in [l.strip().split() for l in archivo] if len(p) >= 5]
                
            predicciones = [[*box.xyxy[0].cpu().numpy()] for r in self.modelo(ruta_imagen, verbose=False)
                    for box in r.boxes if box.conf[0] > 0.25]
            
            for p in predicciones:
                mejor = max([self._iou(p, g) for g in ground_truth], default=0)
                if mejor > 0:
                    ious.append(mejor)

        return np.mean(ious) if ious else 0.0
    
    def _iou(self, b1, b2):
        xi1, yi1, xi2, yi2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        interseccion = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - interseccion
        return interseccion / union if union > 0 else 0.0

    def _evaluar_ocr(self):
        labels = {}

        with open(RUTA_LABELS_OCR, 'r', encoding='utf-8-sig') as archivo:
            for linea in archivo:
                linea = linea.strip()
                partes = linea.split('\t')
                labels[partes[0]] = partes[1]

        lista_imagenes = sorted(f for f in os.listdir(RUTA_IMAGENES_OCR))

        total_cer = 0.0
        total_precision = 0
        contador = 0

        for nombre_imagen in lista_imagenes:

            ruta_imagen = RUTA_IMAGENES_OCR + "/" + nombre_imagen

            imagen = cv2.imread(ruta_imagen)

            texto_pred = self.modelo_ocr.reconocer(imagen)

            texto_gt = labels[nombre_imagen]

            cer = distancia_levenshtein(texto_pred, texto_gt) / max(len(texto_gt), 1)
            total_cer += cer

            if texto_pred == texto_gt:
                total_precision += 1

            contador += 1

        return total_cer / contador, total_precision / contador, contador


    def _medir_latencia_lote(self):
        lista_imagenes = [os.path.join(RUTA_IMAGENES_TEST, f) for f in os.listdir(RUTA_IMAGENES_TEST)[:10]]
        tiempo_inicio = time.time()

        for ruta_imagen in lista_imagenes:
            imagen = cv2.imread(ruta_imagen)
            self.modelo(imagen, verbose=False)

        tiempo_fin = time.time()

        return (tiempo_fin - tiempo_inicio) / len(lista_imagenes) * 1000


    def _medir_latencia_webcam(self):
        captura = cv2.VideoCapture(0)

        if not captura.isOpened():
            return None

        lista_tiempos = []

        for _ in range(100):
            ret, frame = captura.read()

            if not ret:
                break

            inicio = time.time()

            self.modelo(frame, verbose=False)

            fin = time.time()
            lista_tiempos.append((fin - inicio) * 1000)

        captura.release()

        return sum(lista_tiempos) / len(lista_tiempos) if lista_tiempos else None

    def _medir_recursos(self):
        proceso = psutil.Process()
        ram_mb = proceso.memory_info().rss / 1024 / 1024

        if torch.cuda.is_available():
            memoria_gpu = torch.cuda.memory_allocated() / 1024 / 1024
            return ram_mb, memoria_gpu

        return ram_mb, None