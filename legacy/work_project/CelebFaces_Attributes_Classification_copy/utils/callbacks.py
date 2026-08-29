import numpy as np
import torch
import torch.nn as nn
import wandb
from pytorch_lightning.callbacks import Callback
from torchmetrics import F1Score as F1, Accuracy


class MetricsCallback(Callback):
    def __init__(self, num_classes):
        # torchmetrics 1.x: metrics are task-typed; the 40 CelebA attributes are
        # an independent multi-label problem. `compute_on_step`/`average='samples'`
        # were removed — use macro averaging over labels.
        self.acc_mod_val = Accuracy(
            task="multilabel", num_labels=num_classes, average='macro')
        self.F1score_mod_val = F1(
            task="multilabel", num_labels=num_classes, average='macro')
        self.acc_mod_train = Accuracy(
            task="multilabel", num_labels=num_classes, average='macro')
        self.F1score_mod_train = F1(
            task="multilabel", num_labels=num_classes, average='macro')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        _, y = batch
        y = y.cpu()
        outputs['logits'] = outputs['logits'].cpu()
        predictions = torch.round(nn.Sigmoid()(outputs['logits']))

        self.acc_mod_val(predictions, y)
        self.F1score_mod_val(predictions, y)

    def on_validation_epoch_end(self, trainer, pl_module):
        attr_dict = pl_module.attr_dict
        acc = self.acc_mod_val.compute()
        self.acc_mod_val.reset()
        # for i, accuracy in enumerate(acc):
        #    wandb.log({"val/accuracy-{}".format(attr_dict[i]): accuracy})
        # table = wandb.Table(data=acc, columns=attr_dict)
        pl_module.log("val/acc", acc)

        F1 = self.F1score_mod_val.compute()
        self.F1score_mod_val.reset()
        # for i, f_score in enumerate(F1):
        #    wandb.log({"val/F1-scores-{}".format(attr_dict[i]): f_score})
        # table = wandb.Table(data=F1, columns=attr_dict)
        pl_module.log("val/F1", F1)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        _, y = batch
        y = y.cpu()
        outputs['logits'] = outputs['logits'].cpu()
        predictions = torch.round(nn.Sigmoid()(outputs['logits']))

        self.acc_mod_train(predictions, y)
        self.F1score_mod_train(predictions, y)

    def on_train_epoch_end(self, trainer, pl_module):

        acc = self.acc_mod_train.compute()
        self.acc_mod_train.reset()
        pl_module.log("train/acc", acc)

        F1 = self.F1score_mod_train.compute()
        self.F1score_mod_train.reset()
        pl_module.log("train/F1", F1)


class WandbImageCallback(Callback):

    def __init__(self, nb_image):
        self.nb_image = nb_image

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        attr_dict = pl_module.attr_dict
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return  # logger disabled (e.g. fast_dev_run / no wandb) — skip image logging
        if batch_idx == 0:
            x, y = batch
            images = x[:self.nb_image].cpu()
            labels = np.array(y[:self.nb_image].cpu())
            predictions = torch.round(nn.Sigmoid()(
                outputs["logits"][:self.nb_image].cpu()))
            preds = np.array(predictions)

            samples = []
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            for i in range(images.shape[0]):

                bg_image = images[i].detach().numpy().transpose((1, 2, 0))
                bg_image = std * bg_image + mean
                bg_image = np.clip(bg_image, 0, 1)

                predicted_labels = np.where(preds[i] == 1)[0].tolist()
                predicted_labels = [attr_dict[idx] for idx in predicted_labels]

                samples.append(wandb.Image(
                    bg_image, caption=str(predicted_labels)))

            trainer.logger.experiment.log({"val/predictions": samples})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        attr_dict = pl_module.attr_dict
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return  # logger disabled (e.g. fast_dev_run / no wandb) — skip image logging
        if batch_idx == 0:
            x, y = batch
            images = x[:self.nb_image].cpu()
            labels = np.array(y[:self.nb_image].cpu())
            preds = np.array(torch.round(nn.Sigmoid()(
                outputs["logits"][:self.nb_image].cpu())))

            samples = []
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            for i in range(images.shape[0]):

                bg_image = images[i].detach().numpy().transpose((1, 2, 0))
                bg_image = std * bg_image + mean
                bg_image = np.clip(bg_image, 0, 1)

                predicted_labels = np.where(preds[i] == 1)[0].tolist()
                predicted_labels = [attr_dict[idx] for idx in predicted_labels]

                samples.append(wandb.Image(
                    bg_image, caption=str(predicted_labels)))

            trainer.logger.experiment.log({"train/predictions": samples})
