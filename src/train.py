from ultralytics import YOLO

MODELO = "yolov8s.pt"
RUTA_DATOS = "data/data.yaml"
EPOCHS = 20
TAMANIO_IMAGEN = 640
PROYECTO = "runs"
NOMBRE_MODELO = "detector"

def entrenar_modelo():

    modelo = YOLO(MODELO)

    resultados = modelo.train(data=str(RUTA_DATOS), epochs=EPOCHS, imgsz=TAMANIO_IMAGEN, project=PROYECTO,name=NOMBRE_MODELO)

    return resultados

if __name__ == "__main__":
    entrenar_modelo()