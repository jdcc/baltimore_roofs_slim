import random
import string
import time
from copy import deepcopy
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm.auto import tqdm

# from .config import config
from ..data import fetch_image_from_hdf5
from .models import numpy_to_tensor

# TODO put into config
SEED = 1
MODEL_DIR = Path("/tmp")


# TODO Clean up data prep from actual modeling
class ImageModel:
    def __init__(
        self,
        hdf5,
        batch_size,
        learning_rate=None,
        num_epochs=None,
        load_model=None,
        algorithm="AdamW",
        pretrained="ResNet18",
        angle_variations=[0],
        optimizer=None,
        unfreeze=0,
        dropout=0.0,
    ):
        self.hdf5 = hdf5
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.load_model = load_model
        self.pretrained = pretrained
        self.algorithm = algorithm
        self.angle_variations = angle_variations
        self.best_state = None
        self.optimizer = optimizer
        self.unfreeze = unfreeze
        self.dropout = dropout

        random_prefix = "".join(random.sample(string.ascii_letters, 4))
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.model_name = f"{now}_{random_prefix}"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.load_model:
            self.model = torch.load(self.load_model)
            self.model.to(self.device)
        else:
            model_class_name = self.pretrained.lower()
            model_class = getattr(models, model_class_name)
            weights_attr = getattr(models, f"{self.pretrained}_Weights")
            self.model = set_model(
                self.device,
                model_class(weights=weights_attr.DEFAULT),
                self.unfreeze,
                self.dropout,
            )

    def checkpoint(self):
        return {
            "model": deepcopy(self.model.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
        }

    def to_save(self):
        return {
            "best_state": self.best_state,
            "current_state": self.checkpoint(),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "pretrained": self.pretrained,
            "algorithm": self.algorithm,
            "angle_variations": self.angle_variations,
            "optimizer": self.optimizer,
            "model_name": self.model_name,
            "unfreeze": self.unfreeze,
            "dropout": self.dropout,
        }

    @classmethod
    def load(cls, state):
        instance = cls(
            batch_size=state["batch_size"],
            learning_rate=state["learning_rate"],
            num_epochs=state["num_epochs"],
            pretrained=state["pretrained"],
            algorithm=state["algorithm"],
            angle_variations=state["angle_variations"],
            optimizer=state["optimizer"],
            unfreeze=state["unfreeze"],
            dropout=state["dropout"],
        )
        instance.model.load_state_dict(state["best_state"]["model"])
        instance.optimizer.load_state_dict(state["best_state"]["optimizer"])
        return instance

    def get_dataloaders(self, X, y):
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        train_dataloader = get_dataloader(
            train_X,
            train_y,
            self.hdf5,
            self.batch_size,
            angle_variations=self.angle_variations,
            shuffle=True,
        )
        val_dataloader = get_dataloader(
            test_X,
            test_y,
            self.hdf5,
            self.batch_size,
            angle_variations=self.angle_variations,
            shuffle=False,
        )

        return {"train": train_dataloader, "val": val_dataloader}

    def log_to_tensorboard(self, phase, epoch, epoch_loss):
        logdir = str(Path(MODEL_DIR) / "tensorboard" / self.model_name)
        hparam_dict = {
            "lr": self.learning_rate,
            "bsize": self.batch_size,
            "optimizer": self.optimizer.__class__.__name__,
            "pre-trained": self.pretrained,
            "unfreeze": self.unfreeze,
            "dropout": self.dropout,
        }

        metric_dict = {
            "hparam/loss": self.best_loss,
        }

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            w_hp.add_hparams(hparam_dict, metric_dict, run_name=f"/{logdir}")

    def fit(self, X, y):
        assert self.learning_rate is not None
        assert self.num_epochs is not None

        self.best_acc = 0.0
        self.best_loss = float("inf")
        self.criterion = nn.CrossEntropyLoss()
        if self.optimizer is None:
            optim_class = getattr(optim, self.algorithm)
            self.optimizer = optim_class(self.model.parameters(), self.learning_rate)

        dataloaders = self.get_dataloaders(X, y)

        for epoch in tqdm(range(self.num_epochs), desc="Epoch", leave=False):
            for phase in tqdm(["train", "val"], desc="Phase", leave=False):
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                batch_losses = []

                for inputs, labels in tqdm(
                    dataloaders[phase], desc="Batch", leave=False
                ):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                    batch_loss = loss.item()
                    batch_losses.append(batch_loss)

                epoch_loss = np.array(batch_losses).mean()

                if phase == "val":
                    if epoch_loss < self.best_loss:
                        self.best_loss = epoch_loss
                        self.best_state = self.checkpoint()
                    acc = (outputs.argmax(dim=1) == labels).to(torch.float).mean()
                    prec, rec, f1, support = precision_recall_fscore_support(
                        labels.cpu(),
                        outputs.argmax(dim=1).cpu(),
                        average="binary",
                        zero_division=0,
                    )
                    tqdm.write(
                        f"Epoch: {epoch:4}; "
                        f"Validation loss: {epoch_loss:6.4f}; "
                        f"acc: {acc:.2%}; "
                        f"prec: {prec:.2%}; "
                        f"recall: {rec:.2%}; "
                        f"f1: {f1:.2}; "
                        f"support: {'' if support is None else support}; "
                    )

                self.log_to_tensorboard(phase, epoch, epoch_loss)

        return self

    def forward(self, X: list[str]) -> dict[str, float]:
        dataloader = get_dataloader(
            X,
            [0] * len(X),
            self.hdf5,
            self.batch_size,
            angle_variations=[0],
            shuffle=False,
        )

        self.model.eval()
        probs = []

        for inputs, _ in tqdm(dataloader, desc="Batch", leave=False):
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                probs_for_batch = torch.nn.functional.softmax(outputs, dim=1)
                probs.append(probs_for_batch)

        results = torch.cat(probs, dim=0).cpu().numpy()
        if len(X) != len(results):
            raise ValueError(
                f"Number of results ({len(results)}) "
                "does not match number of inputs ({len(X)})"
            )

        output = [float(damage_pred) for damage_pred in results[:, 1]]
        return dict(zip(X, output))

    def predict_proba(self, X: list[str]) -> dict[str, float]:
        return self.forward(X)


def get_dataloader(X, y, hdf5, batch_size, angle_variations=[0], shuffle=False):
    dataset = BlocklotDataset(
        X,
        transform=ImageStandardizer(output_dims=(224, 224)),
        angle_variation=angle_variations,
        labels=y,
        hdf5=hdf5,
    )  # TODO: update transform?
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def set_model(device, pretrained, unfreeze=0, dropout=0.0):
    model = pretrained
    for param in model.parameters():
        param.requires_grad = False

    if unfreeze == 1:
        for param in model.layer4[1].parameters():
            param.requires_grad = True

    if unfreeze == 2:
        for param in model.layer4.parameters():
            param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_ftrs, 2))
    return model.to(device)


class BlocklotDataset(torch.utils.data.Dataset):
    """Characterizes a blocklot dataset for PyTorch"""

    def __init__(
        self,
        blocklot_ids: list[str],
        labels: Mapping[str, float],
        hdf5: h5py.File,
        transform=None,
        angle_variation=[0],
    ):
        self.blocklot_ids = blocklot_ids
        self.len_blocklots = len(self.blocklot_ids)
        self.angle_variations = angle_variation
        self.angles_len = len(self.angle_variations)
        self.angles = self.angle_variations * self.len_blocklots
        self.blocklot_ids = [
            val for val in self.blocklot_ids for _ in range(self.angles_len)
        ]

        self.labels: list[str] = [val for val in labels for _ in range(self.angles_len)]
        if isinstance(hdf5, Path):
            hdf5 = h5py.File(hdf5)
        self.hdf5 = hdf5
        self.transform = transform

    def __len__(self):
        return len(self.blocklot_ids)

    def __getitem__(self, index):
        return self.get_image(index)

    def get_image(self, index):
        blocklot_id = self.blocklot_ids[index]

        image_data = fetch_image_from_hdf5(blocklot_id, self.hdf5)
        label = self.labels[index]

        angle = self.angles[index]

        image_data = transform.rotate(
            numpy_to_tensor(image_data).to(torch.uint8),
            angle,
            expand=True,
            interpolation=transform.InterpolationMode.BILINEAR,
        )

        if self.transform is None:
            return image_data, label

        return self.transform(image_data), label


class InMemoryBlocklotDataset(BlocklotDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = [
            self.get_image(i)
            for i in tqdm(
                range(len(self)), desc="Loading images", leave=False, smoothing=0
            )
        ]

    def __getitem__(self, index):
        return self.dataset[index]


class ImageStandardizer:
    def __init__(self, output_dims=None, pad=True):
        self.output_dims = output_dims
        self.pad = pad

    def __call__(self, x):
        if self.pad:
            max_dim = max(x.shape[1], x.shape[2])
            padding = (int((max_dim - x.shape[2]) / 2), int((max_dim - x.shape[1]) / 2))
            x = transform.pad(x, padding, fill=0)

        if self.output_dims is not None:
            x = transform.resize(
                x,
                self.output_dims,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        x = x.nan_to_num()
        return x.to(torch.float32)


def standardize_tensors(blocklot_to_tensors: dict[str, list[torch.Tensor]]):
    standardizer = ImageStandardizer(pad=False)

    standardized_images = {}

    for blocklot, variants in blocklot_to_tensors.items():
        image_list = []
        for image in variants:
            image_list.append(standardizer(image))
        standardized_images[blocklot] = image_list
    return standardized_images
