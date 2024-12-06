from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import os
import shutil
from tqdm import tqdm


def organize_by_yolo_predictions(yolo_model_path, input_dir, output_base_dir, labels=None):
    model = YOLO(yolo_model_path)
    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)

    no_detection_dir = output_base_path / "no_detection"
    no_detection_dir.mkdir(exist_ok=True)

    image_paths = list(Path(input_dir).glob("*.jpg"))
    stats = {"total": len(image_paths), "processed": 0, "by_class": {}}

    print(f"Procesando {len(image_paths)} imágenes...")

    for img_path in tqdm(image_paths):
        try:
            results = model(str(img_path), verbose=False)[0]

            if len(results.boxes) == 0:
                if labels is None or "no_detection" in labels:
                    shutil.copy2(img_path, no_detection_dir / img_path.name)
                    stats["by_class"]["no_detection"] = stats["by_class"].get("no_detection", 0) + 1
                continue

            confidences = results.boxes.conf.cpu().numpy()
            best_idx = confidences.argmax()
            box = results.boxes[best_idx]

            cls_id = box.cls.item()
            cls_name = results.names[int(cls_id)]

            # Skip if class not in labels
            if labels is not None and cls_name not in labels:
                continue

            confidence = confidences[best_idx]

            class_dir = output_base_path / cls_name
            class_dir.mkdir(exist_ok=True)

            bbox = box.xyxy[0].cpu().numpy()
            original_image = Image.open(img_path)
            cropped_image = original_image.crop((
                int(bbox[0]), int(bbox[1]),
                int(bbox[2]), int(bbox[3])
            ))

            output_path = class_dir / f"{img_path.stem}_{confidence:.2f}.jpg"
            cropped_image.save(output_path)

            stats["by_class"][cls_name] = stats["by_class"].get(cls_name, 0) + 1
            stats["processed"] += 1

        except Exception as e:
            print(f"\nError procesando {img_path.name}: {str(e)}")
            continue

    print("\nProcesamiento completado:")
    print(f"Total imágenes procesadas: {stats['processed']}/{stats['total']}")
    print("\nImágenes por clase:")
    for cls_name, count in stats["by_class"].items():
        print(f"{cls_name}: {count} imágenes")


# Ejemplo de uso
if __name__ == "__main__":
    organize_by_yolo_predictions(
        yolo_model_path="api/best.pt",
        input_dir="dniDorso",
        output_base_dir="dniDorso_processed_by_class",
        labels=None
    )