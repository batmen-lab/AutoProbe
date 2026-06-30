import torch
from pytorch_lightning import LightningModule
import timm
from torch.optim import AdamW
import torch.nn as nn
import numpy as np

from utils.constant import ATTRIBUTES

class Classification(LightningModule):

    def __init__(self, config, attr_dict=None):
        """method used to define our model parameters"""
        super().__init__()

        #self.loss = nn.BCELoss()
        self.loss = nn.BCEWithLogitsLoss()
        self.attr_dict = attr_dict

        # optimizer parameters
        self.lr = config.lr if hasattr(config, 'lr') else None

        self.net = timm.create_model(config.model_name,
                                    pretrained=config.pretrained,
                                    num_classes=config.n_classes,
                                    drop_rate=0.3,
                                    drop_path_rate=0.2)

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        bce_loss, logits = self._get_preds_loss_accuracy(batch, detach_logits=False)

        _, y = batch
        name_to_idx = {v: k for k, v in ATTRIBUTES.items()}
        male_idx = name_to_idx["Male"]
        target_attrs = ["Heavy_Makeup", "Wearing_Lipstick", "Attractive",
                        "Wavy_Hair", "No_Beard", "Arched_Eyebrows"]
        target_idx = [name_to_idx[a] for a in target_attrs]

        probs = torch.sigmoid(logits)
        p_t = probs[:, target_idx]
        m = y[:, male_idx].float()
        p_t_centered = p_t - p_t.mean(dim=0, keepdim=True)
        m_centered = m - m.mean()
        pen = ((p_t_centered * m_centered.unsqueeze(1)).mean(dim=0) ** 2).mean()
        total_loss = bce_loss + 0.5 * pen

        # Log loss and metric
        self.log("train/loss", total_loss)

        return {"loss": total_loss, "logits": logits.detach()}

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val/loss", loss)

        # Let's return preds to use it in a custom callback
        return {"logits": logits}

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("test/loss", loss)

        return {"logits": logits}

    def predict_step(self, batch, batch_idx):
        x, img_name = batch


        logits = self(x)
        converted_logits = nn.Sigmoid()(logits.detach())
        preds = torch.round(converted_logits)

        converted_preds = preds.detach().cpu().numpy()
        batch_converted_preds = []
        for pred_batch in converted_preds:
            batch_converted_preds.append([ATTRIBUTES[i] for i in np.where(pred_batch==1.0)[0]])
        return img_name, preds, batch_converted_preds, converted_logits.cpu().numpy()

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _get_preds_loss_accuracy(self, batch, detach_logits=True):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.float())
        return loss, (logits.detach() if detach_logits else logits)