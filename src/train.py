from ultralytics import YOLO

def entrenar_modelo():

    modelo = YOLO("yolov8s.pt")

    archivo_datos = "data/data.yaml"

    resultados = modelo.train(
        data=str(archivo_datos),
        epochs=20,
        patience=10,
        imgsz=640,
        project="runs",
        name="detector"
    )

    return resultados

if __name__ == "__main__":
    entrenar_modelo()