import os
import cv2


def process_image(image, detector, ocr):
    detections = detector.predict(image)

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        bb_conf = det["conf"]

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        text, ocr_conf = ocr.read(crop)

        if isinstance(text, list):
            text = text[0]

        label = f"{text} | {bb_conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    return image


def run_webcam(detector, ocr):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_image(frame, detector, ocr)
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path, detector, ocr):
    image = cv2.imread(image_path)
    image = process_image(image, detector, ocr)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_folder(folder_path, detector, ocr):
    files = [f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not files:
        print("No hay imágenes en la carpeta.")
        return

    index = 0

    while True:
        path = os.path.join(folder_path, files[index])
        image = cv2.imread(path)

        if image is None:
            index += 1
            continue

        processed = process_image(image, detector, ocr)

        cv2.imshow("Modo carpeta", processed)
        print(f"Imagen {index+1}/{len(files)}")
        print("n = siguiente | b = anterior | q = salir")

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            index = (index + 1) % len(files)
        elif key == ord("b"):
            index = (index - 1) % len(files)

    cv2.destroyAllWindows()

