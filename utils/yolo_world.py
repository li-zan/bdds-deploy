from time import perf_counter
from typing import Any
from PIL import Image

from ultralytics import YOLOWorld as yolo_world

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)


class YOLOWorld:
    def __init__(self, **kwargs):
        self.model = yolo_world(kwargs['ckpt'])
        self.device = kwargs['device']
        self.class_names = None

    def infer(
            self,
            image: Any = None,
            text: list = None,
            confidence: float = 0.3,
            **kwargs,
    ):
        t1 = perf_counter()
        if isinstance(image, Image.Image):
            img_dims = image.size
        else:
            img_dims = image.shape

        self.model.set_classes(text)
        self.class_names = text
        results = self.model.predict(
            image,
            conf=confidence,
            iou=kwargs['iou'],
            device=self.device,
            # verbose=False,
        )

        t2 = perf_counter() - t1

        predictions = []
        for result in results:
            for i, box in enumerate(result.boxes):
                x, y, w, h = box.xywh.tolist()[0]
                class_id = int(box.cls)
                predictions.append(
                    ObjectDetectionPrediction(
                        **{
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "confidence": float(box.conf),
                            "class": self.class_names[class_id],
                            "class_id": class_id,
                        }
                    )
                )

        responses = ObjectDetectionInferenceResponse(
            predictions=predictions,
            image=InferenceResponseImage(width=img_dims[1], height=img_dims[0]),
            time=t2,
        )
        return responses
