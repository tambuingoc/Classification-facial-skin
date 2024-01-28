
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision

from tempfile import TemporaryDirectory
from src.configs.model_config import config


def create_resnet_model(num_classes: int, device: str):
    model = torchvision.models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.85),
        nn.Linear(num_ftrs, 512),
        nn.Dropout(0.7),
        nn.Linear(512, 128),
        nn.Dropout(0.7),
        nn.Linear(128, num_classes)
    )
    model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    return model
