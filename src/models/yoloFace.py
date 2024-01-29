from yolo5face.get_model import get_model


def create_face_model(device: str):
    model = get_model("yolov5n", device = device, min_face = 24)
    return model