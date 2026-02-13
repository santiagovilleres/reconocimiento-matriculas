from ultralytics import YOLO

class DetectorMatrículas:
    def __init__(self, pesos):
        self.modelo = YOLO(pesos)

    def predecir(self, imagen, confianza_minima=0.25):
        resultados = self.modelo(imagen)
        detecciones = []

        for caja in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            confianza = float(caja.conf[0])

            if confianza >= confianza_minima:
                detecciones.append({
                    "xyxy": (x1, y1, x2, y2),
                    "confianza": confianza
                })

        return detecciones