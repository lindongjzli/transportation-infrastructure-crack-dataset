import argparse
import json
import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from model_resnet import resnet50


def main(args):
    """
    Training function
    :param args: training arguments
    :return:
    """
    print(args)
    print(
        'Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/'
    )
    tb_writer = SummaryWriter()

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
        "train": transforms.Compose(
            [
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Dataset root directory
    dataset_root = os.path.abspath(os.path.join(os.getcwd(), args.dataset_root))
    assert os.path.exists(dataset_root), "dataset root: {} does not exist.".format(
        dataset_root
    )

    # Load training dataset
    train_dataset = MyDataset(
        root=dataset_root,
        images_folder=args.images_folder,
        label_file=args.train_file,
        transform=data_transform["train"],
    )

    # Number of training samples
    train_num = len(train_dataset)

    # Load class name to label mapping
    with open(args.classes_indices) as f:
        classes_indices = json.load(f)
    # Swap key-value pairs
    indices_classes = {v: k for k, v in classes_indices.items()}

    # Batch size
    batch_size = args.batch_size
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
    print("Using {} dataloader workers every process".format(nw))

    # Training data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )

    # Load validation dataset
    val_dataset = MyDataset(
        root=dataset_root,
        images_folder=args.images_folder,
        label_file=args.val_file,
        transform=data_transform["val"],
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

    print(
        "using {} images for training, {} images for validation.".format(
            train_num, val_num
        )
    )

    # Directory for this experiment
    # Get current timestamp
    current = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Build weight save directory path
    current_save_dir = os.path.join(args.save_folder, "resnet")
    if not os.path.exists(current_save_dir):
        os.makedirs(current_save_dir)

    # Model
    model_name = "resnet50"
    net = resnet50()
    if args.weights is None or not os.path.exists(args.weights):
        model_weight_path = "pre-weights/resnet50-pre.pth"
    else:
        model_weight_path = args.weights
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(
        model_weight_path
    )
    net.load_state_dict(torch.load(model_weight_path, map_location="cpu"))

    # freeze features weights
    for param in net.parameters():
        param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, args.num_classes)

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    scheduler = None

    # Number of training epochs
    epochs = args.epochs
    acc = 0.0
    best_acc = 0.0

    # Build weight save directory path
    weights_save_dir = current_save_dir
    if not os.path.exists(weights_save_dir):
        os.makedirs(weights_save_dir)

    # Training step counter
    train_steps = len(train_loader)

    # Epoch loop
    for epoch in range(epochs):
        # Train
        mean_loss = train_one_epoch(
            model=net,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Validate
        acc = macro_evaluate(model=net, data_loader=val_loader, device=device)

        # Print
        # print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # Save weights
        if (epoch + 1) % 5 == 0:
            torch.save(
                net.state_dict(),
                os.path.join(weights_save_dir, "epoch_{}.pth").format(epoch),
            )
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), os.path.join(weights_save_dir, "best.pth"))

    # Print in red
    print(
        "\033[1;31;40m"
        + "Finished {} Training, best accuracy: %.3f".format(model_name) % best_acc
        + "\033[0m"
    )

    print("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Parse arguments
    parser.add_argument("--num_classes", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=640, help="img size")
    parser.add_argument("--weights", type=str, default=None, help="model weights path")
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu"
    )
    parser.add_argument("--dataset_root", default="../data", help="Images folder")
    parser.add_argument(
        "--images_folder", default="coco/JPEGImages", help="Images folder"
    )
    parser.add_argument(
        "--train_file",
        default="specific_2_train.txt",
        help="training config file",
    )
    parser.add_argument(
        "--val_file", default="specific_2_val.txt", help="validation config file"
    )
    parser.add_argument(
        "--save_folder", default="weights_specific_2", help="weight save folder"
    )
    parser.add_argument(
        "--classes_indices", default="data/classes_indices_specific_2.json"
    )
    arguments = parser.parse_args()
    main(arguments)
