import argparse
import json
import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import *

from models.resnet import (
    resnet34,
    resnet50,
    resnet101,
    resnext50_32x4d,
    resnext101_32x8d,
)
from models.mobilenet_v3 import mobilenet_v3_small, mobilenet_v3_large
from models.shufflenet import (
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)
from models.densenet import (
    DenseNet,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    load_state_dict,
)
from models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
)
from models.regnet import create_regnet

MODELS = {
    "resnet34": {"model": resnet34, "optimizer": optim.Adam, "lr": 0.0001},
    "resnet50": {"model": resnet50, "optimizer": optim.Adam, "lr": 0.0001},
    "resnet101": {"model": resnet101, "optimizer": optim.Adam, "lr": 0.0001},
    "resnext50_32x4d": {
        "model": resnext50_32x4d,
        "optimizer": optim.Adam,
        "lr": 0.0001,
    },
    "resnext101_32x8d": {
        "model": resnext101_32x8d,
        "optimizer": optim.Adam,
        "lr": 0.0001,
    },
    "mobilenet_v3_small": {
        "model": mobilenet_v3_small,
        "optimizer": optim.Adam,
        "lr": 0.0001,
    },
    "mobilenet_v3_large": {
        "model": mobilenet_v3_large,
        "optimizer": optim.Adam,
        "lr": 0.0001,
    },
    "shufflenet_v2_x0_5": {
        "model": shufflenet_v2_x0_5,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "lrf": 0.1,
    },
    "shufflenet_v2_x1_0": {
        "model": shufflenet_v2_x1_0,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "lrf": 0.1,
    },
    "shufflenet_v2_x1_5": {
        "model": shufflenet_v2_x1_5,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "lrf": 0.1,
    },
    "shufflenet_v2_x2_0": {
        "model": shufflenet_v2_x2_0,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "lrf": 0.1,
    },
    "densenet121": {
        "model": densenet121,
        "optimizer": optim.SGD,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.1,
    },
    "densenet161": {
        "model": densenet161,
        "optimizer": optim.SGD,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.1,
    },
    "densenet169": {
        "model": densenet169,
        "optimizer": optim.SGD,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.1,
    },
    "densenet201": {
        "model": densenet201,
        "optimizer": optim.SGD,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.1,
    },
    "efficientnet_b0": {
        "model": efficientnet_b0,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.01,
    },
    "efficientnet_b1": {
        "model": efficientnet_b1,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.01,
    },
    "efficientnet_b2": {
        "model": efficientnet_b2,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.01,
    },
    "efficientnet_b3": {
        "model": efficientnet_b3,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.01,
    },
    "efficientnet_b4": {
        "model": efficientnet_b4,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.01,
    },
    "efficientnet_b5": {
        "model": efficientnet_b5,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.01,
    },
    "efficientnet_b6": {
        "model": efficientnet_b6,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.01,
    },
    "efficientnet_b7": {
        "model": efficientnet_b7,
        "optimizer": optim.SGD,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lrf": 0.01,
    },
    "regnet": {
        "model": create_regnet,
        "optimizer": optim.SGD,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 5e-5,
        "lrf": 0.01,
    },
}


def main(args):
    print(args)
    # Get total number of GPUs
    num_device = torch.cuda.device_count()
    # Get device index
    device_id = int(args.device[-1])
    # Validate device index
    assert device_id < num_device, "GPU device id is out of range"
    # Check GPU availability
    assert torch.cuda.is_available(), "No available GPU"
    # Get specified GPU device
    device = torch.device(args.device)
    print("using {} device".format(device))

    # Data preprocessing transforms
    data_transform = {
        "val": transforms.Compose(
            [
                transforms.Resize(args.img_size),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    }

    # Load class name to label mapping
    with open("data/classes_indices.json") as f:
        classes_indices = json.load(f)
    # Swap key-value pairs
    indices_classes = {v: k for k, v in classes_indices.items()}

    # Batch size
    batch_size = args.batch_size
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
    print("Using {} dataloader workers every process".format(nw))

    # Dataset root directory
    dataset_root = os.path.abspath(os.path.join(os.getcwd(), "../datasets/VOC2012"))
    # Validation annotation filename
    val_txt = args.val_file
    # Load validation dataset
    val_dataset = MyDataset(
        root=dataset_root, label_file=val_txt, transform=data_transform["val"]
    )
    # Number of validation samples
    val_num = len(val_dataset)
    # Validation data loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )
    print("using {} images for validation.".format(val_num))

    # Models to evaluate
    assert args.model is not None, "Please specify --model"
    model_name = args.model

    print("Start {} eval ...".format(model_name))

    # Load model by name
    model = MODELS[model_name]["model"](num_classes=args.num_classes).to(device)

    # Load corresponding weights
    assert args.weights is not None, "Please specify --weights"
    model_weights_path = args.weights
    assert os.path.exists(model_weights_path), "weights file: '{}' not exist.".format(
        model_weights_path
    )
    checkpoint = torch.load(model_weights_path, map_location="cpu")
    # Load weights strictly
    try:
        model.load_state_dict(checkpoint, strict=True)
    except Exception as e:
        # Abort
        print("weights file: '{}' is not compatible.".format(model_weights_path))
        exit(1)
    else:
        print("Weights loaded successfully.")

    # Initial accuracy
    acc = 0.0

    # Validate
    start_time = datetime.now()
    acc = evaluate(model=model, data_loader=val_loader, device=device)
    end_time = datetime.now()
    # Compute elapsed time
    time_diff = end_time - start_time
    # Extract milliseconds per sample
    milliseconds = (time_diff.total_seconds() * 1000) / val_num

    # Print in red
    print(
        "\033[1;31;40m"
        + "Finished {} evaluating, accuracy: %.3f, elapsed time {} ns".format(
            model_name, milliseconds
        )
        % acc
        + "\033[0m"
    )

    print("Finished eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Parse arguments
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512, help="img size")
    parser.add_argument(
        "--model", type=str, default=None, help="Model intended to be evaluated"
    )
    parser.add_argument("--weights", type=str, default=None, help="model weights path")
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu"
    )
    parser.add_argument(
        "--val_file",
        default="static_close_up_view_val.txt",
        help="validation config file",
    )
    arguments = parser.parse_args()
    main(arguments)
