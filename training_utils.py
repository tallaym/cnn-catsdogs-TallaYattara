import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cudnn.benchmark : accélère les convolutions quand la taille d'entrée est
# fixe (ce qui est le cas ici, toutes les images étant redimensionnées
# pareil), sans coût sur la qualité.
torch.backends.cudnn.benchmark = True

# Mixed precision (AMP) : quasi 2x plus rapide sur GPU, perte de précision
# négligeable. Permet de garder des images en pleine résolution (224x224)
# et/ou plus d'epochs dans le même temps. Automatiquement désactivé si pas
# de GPU (autocast CPU n'apporte rien ici).
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# ------------------------------------------------------------------
# Données
#
# Le chargement des données (ImageFolder, transforms, train_data,
# test_data) est déjà fait dans le notebook Part_7 — on n'y touche pas.
# On ajoute seulement un split train/valid par-dessus le `train_data`
# existant, nécessaire pour le scheduler ReduceLROnPlateau et le
# suivi val_loss/val_acc demandés par le devoir.
# ------------------------------------------------------------------

def make_data_loaders(train_data, test_data, batch_size=32, valid_size=0.15, num_workers=2):
    """
    À partir des datasets déjà construits dans Part_7 (train_data, test_data,
    issus de datasets.ImageFolder avec leurs transforms respectifs), construit
    un dict {"train", "valid", "test"} de DataLoaders, en réservant `valid_size`
    du train pour la validation.

    Ne recharge rien, ne retouche pas les transforms : reprend exactement
    ce que Part_7 a déjà défini.
    """

    n_tot = len(train_data)
    indices = torch.randperm(n_tot)
    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    data_loaders = {
        "train": torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
        ),
        # NOTE : valid est échantillonné depuis train_data, donc avec les
        # transforms (augmentation) définies pour l'entraînement dans Part_7.
        # Si tu veux une validation sans augmentation, définis un second
        # ImageFolder sur le même dossier avec test_transforms, comme fait
        # pour train_data/test_data dans Part_7, et passe-le ici à la place.
        "valid": torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
        ),
        "test": torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, num_workers=num_workers
        ),
    }

    return data_loaders


# ------------------------------------------------------------------
# Entraînement / évaluation
# ------------------------------------------------------------------

def _make_metrics(n_classes):
    return {
        "acc": MulticlassAccuracy(num_classes=n_classes).to(device),
        "prec": MulticlassPrecision(num_classes=n_classes).to(device),
        "rec": MulticlassRecall(num_classes=n_classes).to(device),
    }


def train_one_epoch(train_dataloader, model, optimizer, loss, metrics):
    """
    Performs one epoch of training. Returns (avg_loss, {"acc":..., "prec":..., "rec":...}).
    """

    model.to(device)
    model.train()

    for m in metrics.values():
        m.reset()

    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader), desc="Training",
        total=len(train_dataloader), leave=True, ncols=80,
    ):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(data)
            loss_value = loss(output, target)
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

        preds = output.data.max(1, keepdim=True)[1].squeeze()
        for m in metrics.values():
            m.update(preds, target)

    results = {k: m.compute().item() for k, m in metrics.items()}
    return train_loss, results


def valid_one_epoch(valid_dataloader, model, loss, metrics):
    """
    Validate at the end of one epoch. Returns (avg_loss, {"acc":..., "prec":..., "rec":...}).
    """

    for m in metrics.values():
        m.reset()

    with torch.no_grad():
        model.eval()
        model.to(device)

        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader), desc="Validating",
            total=len(valid_dataloader), leave=True, ncols=80,
        ):
            data, target = data.to(device), target.to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(data)
                loss_value = loss(output, target)

            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

            preds = output.data.max(1, keepdim=True)[1].squeeze()
            for m in metrics.values():
                m.update(preds, target)

    results = {k: m.compute().item() for k, m in metrics.items()}
    return valid_loss, results


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, n_classes):
    """
    Full training loop: trains n_epochs, tracks loss/accuracy/precision/recall
    on train and valid at every epoch, reduces lr on plateau, and saves the
    best model (lowest valid loss) to save_path.

    Returns a history dict with keys:
        train_loss, valid_loss, train_acc, valid_acc,
        train_prec, valid_prec, train_rec, valid_rec
    """

    scheduler = ReduceLROnPlateau(optimizer, "min", threshold=0.01)

    train_metrics = _make_metrics(n_classes)
    valid_metrics = _make_metrics(n_classes)

    valid_loss_min = None
    history = {k: [] for k in [
        "train_loss", "valid_loss", "train_acc", "valid_acc",
        "train_prec", "valid_prec", "train_rec", "valid_rec",
    ]}

    for epoch in range(1, n_epochs + 1):
        train_loss, train_res = train_one_epoch(
            data_loaders["train"], model, optimizer, loss, train_metrics
        )
        valid_loss, valid_res = valid_one_epoch(
            data_loaders["valid"], model, loss, valid_metrics
        )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_res["acc"])
        history["valid_acc"].append(valid_res["acc"])
        history["train_prec"].append(train_res["prec"])
        history["valid_prec"].append(valid_res["prec"])
        history["train_rec"].append(train_res["rec"])
        history["valid_rec"].append(valid_res["rec"])

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_res['acc']:.4f} | "
              f"valid loss {valid_loss:.4f} acc {valid_res['acc']:.4f} "
              f"prec {valid_res['prec']:.4f} rec {valid_res['rec']:.4f}")

        if valid_loss_min is None or ((valid_loss_min - valid_loss) / valid_loss_min > 0.01):
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

        scheduler.step(valid_loss)

    return history


def test_model(test_dataloader, model, loss):
    """
    Evaluate on the test set. Returns (test_loss, preds, actuals).
    """

    test_loss = 0.0
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        model.eval()
        model.to(device)

        preds_all = []
        actuals_all = []

        for batch_idx, (data, target) in tqdm(
            enumerate(test_dataloader), desc="Testing",
            total=len(test_dataloader), leave=True, ncols=80,
        ):
            data, target = data.to(device), target.to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(data)
                loss_value = loss(logits, target).detach()

            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            pred = logits.data.max(1, keepdim=True)[1]
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

            preds_all.extend(pred.data.cpu().numpy().squeeze())
            actuals_all.extend(target.data.view_as(pred).cpu().numpy().squeeze())

    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {100. * correct / total:.2f}% ({int(correct)}/{int(total)})")

    return test_loss, preds_all, actuals_all


def plot_confusion_matrix(pred, truth, classes):
    gt = pd.Series(truth, name="Ground Truth")
    predicted = pd.Series(pred, name="Predicted")

    confusion_matrix = pd.crosstab(gt, predicted)
    confusion_matrix.index = classes
    confusion_matrix.columns = classes

    fig, sub = plt.subplots()
    with sns.plotting_context("notebook"):
        ax = sns.heatmap(
            confusion_matrix, annot=True, fmt="d", ax=sub,
            linewidths=0.5, linecolor="lightgray", cbar=False,
        )
        ax.set_xlabel("truth")
        ax.set_ylabel("pred")

    return confusion_matrix


# ------------------------------------------------------------------
# LR finder
# ------------------------------------------------------------------

def lr_finder(min_lr, max_lr, n_steps, loss, model, data_loaders, skip_start=15):
    """
    Ramps the learning rate exponentially from min_lr to max_lr over
    n_steps batches and records the loss at each step, to help pick a
    good learning rate. Restores the model's initial weights afterwards.

    skip_start : nombre de steps initiaux pendant lesquels le critère de
    divergence est désactivé. L'EMA (beta=0.98) n'a une fenêtre effective
    fiable qu'après environ 1/(1-beta) = 50 steps ; avant ça, elle reste
    proche d'une moyenne des toutes premières valeurs et peut déclencher
    un arrêt prématuré à cause du simple bruit batch-à-batch (surtout avec
    un petit batch_size et BatchNorm), sans rapport avec le LR lui-même.
    """

    model.to(device)

    backup_path = "__weights_backup.pt"
    torch.save(model.state_dict(), backup_path)

    optimizer = optim.SGD(model.parameters(), lr=min_lr)

    r = np.power(max_lr / min_lr, 1 / (n_steps - 1))

    def new_lr(step):
        return r ** step

    lr_scheduler = LambdaLR(optimizer, new_lr)

    model.train()

    # Lissage EMA (moyenne mobile exponentielle) avec correction de biais,
    # comme dans fast.ai, plutôt qu'une moyenne cumulée depuis le début.
    # Avec la moyenne cumulée, le tout premier batch pèse énormément sur
    # min(losses.values()) : une loss anormalement basse sur ce batch (bruit,
    # BatchNorm pas encore stabilisé) suffit à déclencher le critère de
    # divergence (>10x le minimum) quelques steps plus tard, alors que le LR
    # est encore minuscule. L'EMA "oublie" progressivement les tout premiers
    # batches et donne une courbe bien plus exploitable.
    beta = 0.98
    avg_loss = 0.0
    best_loss = None

    losses = {}
    n = 0
    keep_going = True

    while keep_going:
        for data, target in tqdm(
            data_loaders["train"], desc="Learning rate finder",
            total=len(data_loaders["train"]), leave=True, ncols=80,
        ):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss_value = loss(output, target)
            loss_value.backward()
            optimizer.step()

            avg_loss = beta * avg_loss + (1 - beta) * loss_value.data.item()
            smoothed_loss = avg_loss / (1 - beta ** (n + 1))  # correction de biais
            losses[lr_scheduler.get_last_lr()[0]] = smoothed_loss

            if best_loss is None or smoothed_loss < best_loss:
                best_loss = smoothed_loss

            if n >= skip_start and smoothed_loss / best_loss > 10:
                keep_going = False
                break

            if n == n_steps - 1:
                keep_going = False
                break

            lr_scheduler.step()
            n += 1

    model.load_state_dict(torch.load(backup_path))
    os.remove(backup_path)

    return losses
