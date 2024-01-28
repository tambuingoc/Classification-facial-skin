import numpy as np
from src.models.resnet import create_resnet_model
from src.configs.model_config import config
from src.utils.model import load_model


def process_model1(image):
    # create model
    model = create_resnet_model(num_classes=3, device=config.device)
    # load best model
    load_model(model, config.model1_path)
    # eval model
    model.eval()

    # output
    output = model(image)

    # get prediction
    class_names = ['cAverage', 'cFair', 'cGood']

    return class_names[output.argmax()]
