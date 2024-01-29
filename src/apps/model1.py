import numpy as np
from src.models.resnet import create_resnet_model
from src.models.mobile import create_mobile_model
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

def process_modelEye(image):
    #create model
    model = create_resnet_model(num_classes=3, device=config.device)
    #load moel forehead best
    load_model(model, config.modelEye_path)
    model.eval()
    output = model(image)
    class_names = ['cAverage', 'cFair', 'cGood']
    
    return class_names[output.argmax()]

def process_modelFore(image):
    #create model
    model = create_resnet_model(num_classes=3, device=config.device)
    #load moel forehead best
    load_model(model, config.modelFore_path)
    model.eval()
    output = model(image)
    class_names = ['cAverage', 'cFair', 'cGood']
    
    return class_names[output.argmax()]

def process_modelSmile(image):
    #create model
    model = create_resnet_model(num_classes=3, device=config.device)
    #load moel Smilehead best
    load_model(model, config.modelSmile_path)
    model.eval()
    output = model(image)
    class_names = ['cAverage', 'cFair', 'cGood']
    
    return class_names[output.argmax()]

def process_modelPig(image):
    model = create_mobile_model(num_classes=3, device=config.device)
    load_model(model, config.modelPig_path)
    model.eval()
    output = model(image)
    class_names = ['cAverage', 'cFair', 'cGood']
    return class_names[output.argmax()]

def process_modelPore(image):
    model = create_mobile_model(num_classes=3, device=config.device)
    load_model(model, config.modelPore_path)
    model.eval()
    output = model(image)
    class_names = ['cAverage', 'cFair', 'cGood']
    return class_names[output.argmax()]
