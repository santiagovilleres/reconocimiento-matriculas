from ultralytics import YOLO

def train_model():

    model = YOLO("yolov8s.pt")

    data_yaml = "data/data.yaml"

    results = model.train(
        data=str(data_yaml),
        epochs=20,
        patience=10,
        imgsz=640,
        project="runs",
        name="detector"
    )

    return results

if __name__ == "__main__":
    train_model()