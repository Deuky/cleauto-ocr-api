from .crnn import crnn
from . import config
import torch
import cv2
import numpy

class OCR():
    _model = None

    def __init__(self):
        self.height = config.IMAGE_HEIGHT
        self.classes = config.N_CLASSES
        self.channels = config.N_CHANNELS
        self.device = config.DEVICE
        self.model_file = config.MODEL_FILE
        self.idx_to_char = config.IDX_TO_CHAR

    def init_model(self):
        if OCR._model is None:
            model = crnn(
                img_h=self.height,
                n_channels=self.channels,
                n_classes=self.classes)
            model.load_state_dict(
                torch.load(
                    self.model_file,
                    map_location=self.device
                )
            )

            model.to(self.device)
            model.eval()

            OCR._model = model

        return OCR._model

    def prediction(self, file):
        model = OCR._model
        img_tensor = self.preprocess_image(file)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=2)
            vin_pred = self.decode_prediction(pred)

        return vin_pred

    def decode_prediction(self, pred):
        pred = pred.squeeze(0).cpu().numpy()

        decoded = []
        prev = None

        for p in pred:
            if p != prev and p != 0:
                decoded.append(self.idx_to_char.get(p, ""))
            prev = p

        return "".join(decoded)

    def preprocess_image(self, file):
        nparr = numpy.frombuffer(file, numpy.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise Exception("Impossible de charger l'image")

        ratio = image.shape[1] / image.shape[0]
        width = int(self.height * ratio)

        image = cv2.resize(
            image, 
            (width, self.height), 
            interpolation=cv2.INTER_AREA
        )
        # img = cv2.GaussianBlur(img, (3,3), 0)
        image = image.astype(numpy.float32) / 255.0 
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)

        return image.to(self.device)
