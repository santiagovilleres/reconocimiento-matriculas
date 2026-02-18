from fast_plate_ocr import LicensePlateRecognizer
import cv2

ANCHO_OCR = 128
ALTO_OCR = 64

class OCR:

    def __init__(self, nombre_modelo):
        self.ocr = LicensePlateRecognizer(nombre_modelo)

    def reconocer(self, imagen_matricula):
        imagen_matricula_preprocesada = self._preprocesar(imagen_matricula)
        texto = self.ocr.run(imagen_matricula_preprocesada)
        return self._postprocesar(texto)


    def _preprocesar(self, imagen):
        imagen_preprocesada= cv2.resize(imagen, (ANCHO_OCR, ALTO_OCR))
        return imagen_preprocesada
    
    
    def _postprocesar(self, texto):
        texto_postprocesado = texto[0].replace("_", "")
        return texto_postprocesado