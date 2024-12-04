import base64
import io

from PIL.Image import Image
from ultralytics import YOLO
import torch


def detect_objects(img_base64: str):

    def load_image_from_base64(base64_str: str) -> torch.Tensor:
        try:
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Error decoding base64 image: {str(e)}")

    image = load_image_from_base64(img_base64)

    yolo = YOLO("YOLODataset/yolov8m.pt")

    predict = yolo.predict(image, imgsz = 640, conf = 0.80)

    boxes = predict[0].boxes
    x1, y1, x2, y2 = boxes.xyxy[0].cpu().numpy()

    return boxes.conf.cpu().numpy(), boxes.names[0]