from ultralytics import YOLO
import cv2
from ocr import OCR
import os

UMBRAL_CONFIANZA = 0.65
EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png")

class Detector:

    def __init__(self, ruta_modelo, nombre_modelo_ocr, umbral_confianza=UMBRAL_CONFIANZA):
        self.modelo = YOLO(ruta_modelo)
        self.ocr = OCR(nombre_modelo_ocr)
        self.umbral_confianza = umbral_confianza


    def procesar_imagen(self, ruta):

        if os.path.isdir(ruta):
            for archivo in os.listdir(ruta):
                if archivo.lower().endswith(EXTENSIONES_IMAGEN):
                    ruta_completa = os.path.join(ruta, archivo)
                    self._procesar_archivo(ruta_completa)
        else:
            self._procesar_archivo(ruta)


    def procesar_webcam(self):

        captura = cv2.VideoCapture(0)

        if not captura.isOpened():
            print("No se pudo acceder a la webcam")
            return

        while True:
            ret, cuadro = captura.read()
            if not ret:
                break

            resultados = self.modelo(cuadro, verbose=False)

            if resultados[0].boxes:
                self._procesar_detecciones(cuadro, resultados)

            cv2.imshow("Webcam", cuadro)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        captura.release()
        cv2.destroyAllWindows()


    def _procesar_archivo(self, ruta_imagen):

        imagen = self._cargar_imagen(ruta_imagen)

        resultados = self.modelo(imagen, verbose=False)

        if resultados[0].boxes:
            self._procesar_detecciones(imagen, resultados)
        else:
            print("No se detectaron matrículas en la imagen")

        cv2.imshow("Imagen", imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def _cargar_imagen(self, ruta):

        imagen = cv2.imread(ruta)

        if imagen is None:
            raise ValueError("No se pudo cargar la imagen")
        return imagen


    def _procesar_detecciones(self, imagen, resultados):

        for resultado in resultados:

            indices_matriculas = (resultado.boxes.cls == 0).nonzero(as_tuple=True)[0]

            for idx in indices_matriculas:

                confianza_yolo = resultado.boxes.conf[idx].item()

                if confianza_yolo > self.umbral_confianza:

                    xyxy = resultado.boxes.xyxy[idx].squeeze().tolist()
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    x2, y2 = int(xyxy[2]), int(xyxy[3])

                    imagen_matricula = imagen[y1:y2, x1:x2]
                    
                    texto = self.ocr.reconocer(imagen_matricula)

                    print(f"Matricula detectada: {texto}")
                    print(f"Confianza YOLO: {confianza_yolo:.2f}")

                    self._dibujar_resultados(imagen, x1, y1, x2, y2, texto)


    def _dibujar_resultados(self, imagen, x1, y1, x2, y2, texto):

        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala = 1.0
        grosor = 2

        (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, fuente, escala, grosor)

        x_inicio = x1
        y_inicio = y1 - alto_texto - 10
        x_fin = x1 + ancho_texto + 10
        y_fin = y1

        cv2.rectangle(imagen, (x_inicio, y_inicio), (x_fin, y_fin), (0, 255, 0), -1)

        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(imagen,texto, (x1 + 5, y1 - 5),fuente, escala, (0, 0, 0),grosor)