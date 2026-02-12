from fast_plate_ocr import LicensePlateRecognizer


class FastPlateOCR:
    def __init__(self, model_name="cct-xs-v1-global-model"):
        self.model = LicensePlateRecognizer(model_name)

    def read(self, image):
        text, conf = self.model.run(image, return_confidence=True)
        return text, conf
