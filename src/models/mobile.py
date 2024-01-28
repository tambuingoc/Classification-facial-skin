
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from tempfile import TemporaryDirectory
from src.configs.model_config import config


def create_mobile_model(num_classes: int, device: str):
    model = models.mobilenet_v2(weights='DEFAULT')
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.7),
        nn.Linear(num_ftrs, 50),
        nn.Dropout(0.5),
        nn.Linear(50, num_classes)
    )
    model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    return model
