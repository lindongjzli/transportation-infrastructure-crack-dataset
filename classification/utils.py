import os.path
import sys
import json
import math
import time
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def parse_txt_annotation(file_path: str) -> list:
    """
    Parse a txt annotation file into a list of dicts, each with keys 'image' and 'label'.
    :param file_path: path to the txt annotation file
    :return:
    """
    data = list()
    # Read file
    with open(file_path, "r") as f:
        for line in f.readlines():
            # Parse each line
            image_name, label = line.strip().split(" ")
            # Build image-label pair
            data.append({"image": image_name, "label": int(label)})

    return data


class MyDataset(Dataset):

    def __init__(self, root, images_folder: str, label_file: str, transform=None):
        """
        Custom dataset class. Label files are stored under dataset/classification_config/, images under dataset/JPEGImages/.
        :param root: dataset root directory
        :param label_file: label filename, each line: <image_name> <class_id>
        :param transform: data transforms
        """
        # Dataset root
        self.root = root
        # Data transforms
        self.transform = transform
        # Image folder
        self.images_folder = images_folder
        # Label file path
        self.label_file_path = os.path.join(root, "classification_config", label_file)
        assert os.path.exists(
            self.label_file_path
        ), f"{self.label_file_path} not exists"
        # Load annotation data
        self.images_frame = parse_txt_annotation(self.label_file_path)

    def __len__(self):
        """
        Return dataset size
        :return:
        """
        return len(self.images_frame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_path = os.path.join(
            self.root, self.images_folder, self.images_frame[index]["image"]
        )
        image = Image.open(image_path).convert("RGB")
        label = self.images_frame[index]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def collate_fn(batch):
        # See PyTorch official default_collate:
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train for one epoch
    :param model: model
    :param optimizer: optimizer
    :param data_loader: data loader
    :param device: device
    :param epoch: epoch index
    :return:
    """
    # Set to training mode
    model.train()
    # Define loss function
    loss_function = torch.nn.CrossEntropyLoss()
    # Running mean loss
    mean_loss = torch.zeros(1).to(device)
    # Zero gradients
    optimizer.zero_grad()

    # Progress bar
    data_loader = tqdm(data_loader, file=sys.stdout)

    # Iterate over data
    for step, data in enumerate(data_loader):
        # Images and labels
        images, labels = data

        # Forward pass
        pred = model(images.to(device))

        # Compute loss
        loss = loss_function(pred, labels.to(device))
        # Backward pass
        loss.backward()
        # Update mean loss
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        data_loader.desc = "[epoch {}] mean loss {}".format(
            epoch, round(mean_loss.item(), 3)
        )

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        # Update parameters
        optimizer.step()
        # Zero gradients
        optimizer.zero_grad()

    return mean_loss.item()


def train_one_epoch_(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss
    accu_num = torch.zeros(1).to(device)  # Accumulated correct predictions
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = (
            "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"],
            )
        )

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Validate for one epoch.
    :param model: model
    :param data_loader: data loader
    :param device: device
    :return:
    """
    # Set to evaluation mode
    model.eval()

    # Total number of validation samples
    total_num = len(data_loader.dataset)

    # Accumulate correct predictions
    sum_num = torch.zeros(1).to(device)

    # Progress bar
    data_loader = tqdm(data_loader, file=sys.stdout)

    # Iterate over data
    for step, data in enumerate(data_loader):
        # Images and labels
        images, labels = data
        # Forward pass
        pred = model(images.to(device))
        # Get predicted class
        pred = torch.max(pred, dim=1)[1]
        # Count correct predictions
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # Return accuracy
    return sum_num.item() / total_num


@torch.no_grad()
def evaluate_(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # Accumulated correct predictions
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def create_lr_scheduler(
    optimizer,
    num_step: int,
    epochs: int,
    warmup=True,
    warmup_epochs=1,
    warmup_factor=1e-3,
    end_factor=1e-6,
):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        Return a learning rate scale factor based on the current step.
        Note: PyTorch calls lr_scheduler.step() once before training starts.
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # Warmup phase: lr scale increases from warmup_factor to 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = x - warmup_epochs * num_step
            cosine_steps = (epochs - warmup_epochs) * num_step
            # Post-warmup: lr scale decreases from 1 to end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (
                1 - end_factor
            ) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # Record trainable parameters
    parameter_group_vars = {
        "decay": {"params": [], "weight_decay": weight_decay},
        "no_decay": {"params": [], "weight_decay": 0.0},
    }

    # Record corresponding parameter names
    parameter_group_names = {
        "decay": {"params": [], "weight_decay": weight_decay},
        "no_decay": {"params": [], "weight_decay": 0.0},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def macro_evaluate(model, data_loader, device):
    """
    Evaluate a multi-class model and return:
      - macro-averaged accuracy
      - macro-averaged precision
      - macro-averaged recall
      - macro-averaged F1-score
      - FPS
      - peak GPU memory usage
    :param model: model
    :param data_loader: validation data loader
    :param device: compute device
    """
    model.eval()  # Set to evaluation mode
    # Store all predictions and labels
    y_pred, y_true = [], []
    # Total number of validation samples
    total_num = len(data_loader.dataset)
    # Timing
    total_time_pure = 0
    # Record start time
    start_time = time.time()
    # Reset peak GPU memory stats
    torch.cuda.reset_peak_memory_stats(device)
    # Validation loop
    with torch.no_grad():
        data_loader_tqdm = tqdm(data_loader, desc="Evaluating", file=sys.stdout)
        for step, (images, labels) in enumerate(data_loader_tqdm):
            images = images.to(device)
            labels = labels.to(device)
            # Record inference start time (for pure inference FPS)
            start_time_pure = time.time()
            # Model inference
            outputs = model(images)  # [batch_size, num_classes]
            # Get predicted class
            preds = torch.argmax(outputs, dim=1)
            # Record inference end time (for end-to-end FPS)
            end_time_pure = time.time()
            # Record per-batch time
            total_time_pure += end_time_pure - start_time_pure

            # Collect on CPU for metric computation
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    # Record end time
    end_time = time.time()
    # ========== 1. Compute FPS ==========
    pure_fps = total_num / total_time_pure
    total_time_end_to_end = end_time - start_time
    end_to_end_fps = total_num / total_time_end_to_end

    # ========== 2. Compute peak GPU memory ==========
    if device.type == "cuda":
        max_mem_allocated = torch.cuda.max_memory_allocated(device) / (
            1024**2
        )  # Convert to MB
    else:
        max_mem_allocated = 0.0

    # ========== 3. Build confusion matrix ==========
    #    cm[i, j]: number of samples of true class i predicted as class j
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    # ========== 4. Compute Macro Accuracy ==========
    # For each class i:
    #   TP_i = cm[i, i]
    #   FN_i = sum(cm[i, :]) - TP_i
    #   FP_i = sum(cm[:, i]) - TP_i
    #   TN_i = total_num - (TP_i + FN_i + FP_i)
    # per-class accuracy_i = (TP_i + TN_i) / total_num
    # Average over all classes to get Macro Accuracy.
    macro_acc_list = []
    for i in range(num_classes):
        TP_i = cm[i, i]
        FN_i = cm[i, :].sum() - TP_i
        FP_i = cm[:, i].sum() - TP_i
        TN_i = total_num - (TP_i + FN_i + FP_i)

        acc_i = (TP_i + TN_i) / total_num
        macro_acc_list.append(acc_i)

    macro_accuracy = float(np.mean(macro_acc_list))
    # ========== 5. Compute Macro Precision / Recall / F1 ==========
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return macro_accuracy, macro_precision, macro_recall, macro_f1, pure_fps, end_to_end_fps, max_mem_allocated
