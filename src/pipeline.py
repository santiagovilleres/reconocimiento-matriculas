import os
import numpy as np
import cv2
import utils

def procesar_imagen(imagen, detector, lector):
    detecciones = detector.predecir(imagen)

    for detección in detecciones:
        x1, y1, x2, y2 = detección["xyxy"]
        confianza_caja = detección["confianza"]

        recorte = imagen[y1:y2, x1:x2]
        if recorte.size == 0:
            continue

        recorte_procesado = utils.pre_procesar(recorte)
        cv2.imshow("Recorte OCR", recorte_procesado)

        texto, confianza_ocr = lector.leer(recorte_procesado)

        if isinstance(texto, list):
            texto = texto[0].replace("_", "")

        if confianza_ocr is None:
            confianza_ocr = 0.0
        else:
            confianza_array = np.array(confianza_ocr, dtype=float)
            confianza_ocr = float(np.mean(confianza_array)) if confianza_array.size > 0 else 0.0

        etiqueta = f"Matricula: {texto} |Conf. YOLO: {confianza_caja:.2f} |Conf. OCR: {confianza_ocr:.2f}"

        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(imagen, etiqueta, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        print (f"Detecciones: {etiqueta}")
    return imagen


def ejecutar_webcam(detector, lector):
    captura = cv2.VideoCapture(0)

    while True:
        exito, fotograma = captura.read()
        if not exito:
            break

        fotograma = procesar_imagen(fotograma, detector, lector)
        cv2.imshow("Webcam", fotograma)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    cv2.destroyWindow("Webcam")


def ejecutar_imagen(ruta_imagen, detector, lector):
    imagen = cv2.imread(ruta_imagen)
    imagen = procesar_imagen(imagen, detector, lector)

    cv2.imshow("Imagen", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ejecutar_carpeta(ruta_carpeta, detector, lector):
    archivos = [f for f in os.listdir(ruta_carpeta)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not archivos:
        print("No hay imágenes en la carpeta.")
        return

    indice = 0

    while True:
        ruta = os.path.join(ruta_carpeta, archivos[indice])
        imagen = cv2.imread(ruta)

        if imagen is None:
            indice += 1
            continue

        procesada = procesar_imagen(imagen, detector, lector)

        cv2.imshow("Modo carpeta", procesada)
        print(f"Imagen {indice+1}/{len(archivos)}")
        print("n = siguiente | b = anterior | q = salir")

        tecla = cv2.waitKey(0) & 0xFF

        if tecla == ord("q"):
            break
        elif tecla == ord("n"):
            indice = (indice + 1) % len(archivos)
        elif tecla == ord("b"):
            indice = (indice - 1) % len(archivos)

    cv2.destroyWindow("Modo carpeta")

