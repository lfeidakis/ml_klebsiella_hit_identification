import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from sklearn.metrics import (
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
)
from sklearn.metrics import precision_score, recall_score

# from models.classifier import AMR_Classifier


def balanced_loss_function():
    return 


class Classifier_Experiment(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        # self.model = AMR_Classifier(config)
        self.model = model
        #self.loss_function = nn.BCEWithLogitsLoss()
        self.threshold = config.get("threshold", 0.5)
        self.test_predictions = []

        if "pos_weight" in config and config["pos_weight"] is not None:
            pos_weight = torch.tensor([config["pos_weight"]], dtype=torch.float32)
            self.loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_function = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        # scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=self.learning_rate, mode="triangular2", step_size_up=10, cycle_momentum=False)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer

    def _step(self, batch):
        # species_idx, spectrum, fprint_tensor, response, dataset = batch
        response = batch[-2]
        logits = self.model(batch)
        loss = self.loss_function(logits.view(-1, 1), response.view(-1, 1))
        # predictions = logits
        predictions = torch.sigmoid(logits)
        predicted_classes = (predictions >= self.threshold).int().cpu().numpy()
        response_classes = response.cpu().numpy()
        logs = {
             "mcc": matthews_corrcoef(response_classes, predicted_classes)}#,
        #     "balanced_accuracy": balanced_accuracy_score(
        #         response_classes, predicted_classes
        #     ),
        #     "f1": f1_score(response_classes, predicted_classes, zero_division=0),
        #     "AUPRC": average_precision_score(
        #         response_classes, logits.cpu().detach().numpy()
        #     ),
        #     "precision": precision_score(
        #         response_classes, predicted_classes, zero_division=0
        #     ),
        #     "recall": recall_score(
        #         response_classes, predicted_classes, zero_division=0
        #     ),
        # }
        #logs = {"dummy": 0}
        return loss, logs, predictions

    def training_step(self, batch, batch_idx):
        loss, logs, predictions = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.batch_size)
        for k, v in logs.items():
            self.log(
                "train_" + k,
                v,
                on_step=False,
                on_epoch=True,
                batch_size=self.batch_size,
            )
        logs["loss"] = loss
        return logs

    def validation_step(self, batch, batch_idx):
        loss, logs, predictions = self._step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.batch_size)
        for k, v in logs.items():
            self.log(
                "val_" + k, v, on_step=False, on_epoch=True, batch_size=self.batch_size
            )
        logs["loss"] = loss
        return logs

    def test_step(self, batch, batch_idx):
        response = batch[-2]
        loss, logs, predictions = self._step(batch)
        # predictions = (predictions >= self.threshold).cpu().int().cpu().flatten().tolist()
        self.test_predictions.extend(predictions.cpu().flatten().tolist())
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        for k, v in logs.items():
            self.log(
                "test_" + k, v, on_step=False, on_epoch=True, batch_size=self.batch_size
            )
        logs["loss"] = loss
        return logs



class Classifier_Experiment_TestMetrics(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        # self.model = AMR_Classifier(config)
        self.model = model
        self.loss_function = nn.BCEWithLogitsLoss()
        self.threshold = config.get("threshold", 0.5)
        self.test_predictions = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        # scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=self.learning_rate, mode="triangular2", step_size_up=10, cycle_momentum=False)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer

    def _step(self, batch):
        # species_idx, spectrum, fprint_tensor, response, dataset = batch
        response = batch[-2]
        logits = self.model(batch)
        loss = self.loss_function(logits.view(-1, 1), response.view(-1, 1))
        # predictions = logits
        predictions = torch.sigmoid(logits)
        # predicted_classes = (predictions >= self.threshold).int().cpu().numpy()
        # response_classes = response.cpu().numpy()
        # logs = {
        #     "mcc": matthews_corrcoef(response_classes, predicted_classes),
        #     "balanced_accuracy": balanced_accuracy_score(
        #         response_classes, predicted_classes
        #     ),
        #     "f1": f1_score(response_classes, predicted_classes, zero_division=0),
        #     "AUPRC": average_precision_score(
        #         response_classes, logits.cpu().detach().numpy()
        #     ),
        #     "precision": precision_score(
        #         response_classes, predicted_classes, zero_division=0
        #     ),
        #     "recall": recall_score(
        #         response_classes, predicted_classes, zero_division=0
        #     ),
        # }
        logs = {"dummy": 0}
        return loss, logs, predictions


    def _test_step(self, batch):
        # species_idx, spectrum, fprint_tensor, response, dataset = batch
        response = batch[-2]
        logits = self.model(batch)
        loss = self.loss_function(logits.view(-1, 1), response.view(-1, 1))
        # predictions = logits
        predictions = torch.sigmoid(logits)
        predicted_classes = (predictions >= self.threshold).int().cpu().numpy()
        response_classes = response.cpu().numpy()
        logs = {
            "mcc": matthews_corrcoef(response_classes, predicted_classes),
            "balanced_accuracy": balanced_accuracy_score(
                response_classes, predicted_classes
            ),
            "f1": f1_score(response_classes, predicted_classes, zero_division=0),
            "AUPRC": average_precision_score(
                response_classes, logits.cpu().detach().numpy()
            ),
            "precision": precision_score(
                response_classes, predicted_classes, zero_division=0
            ),
            "recall": recall_score(
                response_classes, predicted_classes, zero_division=0
            ),
        }
        # logs = {"dummy": 0}
        return loss, logs, predictions

    def training_step(self, batch, batch_idx):
        loss, logs, predictions = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.batch_size)
        for k, v in logs.items():
            self.log(
                "train_" + k,
                v,
                on_step=False,
                on_epoch=True,
                batch_size=self.batch_size,
            )
        logs["loss"] = loss
        return logs

    def validation_step(self, batch, batch_idx):
        loss, logs, predictions = self._step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.batch_size)
        for k, v in logs.items():
            self.log(
                "val_" + k, v, on_step=False, on_epoch=True, batch_size=self.batch_size
            )
        logs["loss"] = loss
        return logs

    def test_step(self, batch, batch_idx):
        response = batch[-2]
        loss, logs, predictions = self._test_step(batch)
        # predictions = (predictions >= self.threshold).cpu().int().cpu().flatten().tolist()
        self.test_predictions.extend(predictions.cpu().flatten().tolist())
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        for k, v in logs.items():
            self.log(
                "test_" + k, v, on_step=False, on_epoch=True, batch_size=self.batch_size
            )
        logs["loss"] = loss
        return logs
