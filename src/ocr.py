from fast_plate_ocr import LicensePlateRecognizer


class LectorMatrícula:
    def __init__(self, nombre_modelo="cct-xs-v1-global-model"):
        self.modelo = LicensePlateRecognizer(nombre_modelo)

    def leer(self, imagen):
        texto, confianza = self.modelo.run(imagen, return_confidence=True)
        return texto, confianza
