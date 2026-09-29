from ultralytics import YOLO


class RoadDamageDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image, confidence=0.05, iou=0.45):
        results = self.model(
            image,
            conf=confidence,
            iou=iou,
            verbose=False
        )

        return results[0]

    def get_detections(self, result):
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                class_name = self.model.names.get(
                    class_id,
                    f"Class_{class_id}"
                )

                detections.append({
                    "class": class_name,
                    "confidence": confidence
                })

        return detections

    def annotate(self, result):
        return result.plot()

