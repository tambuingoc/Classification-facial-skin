from yolo5face.get_model import get_model
import cv2
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize the model
model = get_model("yolov5n", device=device, min_face=24)

# Load your image
image = cv2.imread("uploads/0.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Standard Detection
boxes, key_points, scores = model(image, target_size=512)

for box in boxes:
    [x, y, w, h] = box
    face_crop = image[y:y+h, x:x+w]

    cv2.imshow("crop",face_crop)
    cv2.waitKey(0)