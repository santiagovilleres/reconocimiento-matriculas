from ultralytics import YOLO

class Detector:
    def __init__(self, weights):
        self.model = YOLO(weights)

    def predict(self, image, conf=0.25):
        results = self.model(image)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            if confidence >= conf:
                detections.append({
                    "xyxy": (x1, y1, x2, y2),
                    "conf": confidence
                })

        return detections