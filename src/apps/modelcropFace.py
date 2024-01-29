from src.models.yoloFace import create_face_model
from src.configs.model_config import config
import cv2


def process_crop_face(image):
    model = create_face_model(device=config.device)

    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Standard Detection
    boxes, key_points, scores = model(image, target_size=640)

    if len(boxes) == 0:
        return image, image

    for box in boxes:
        [x, y, w, h] = box
        face_crop = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        break

    return image, face_crop
