import torch
import torch.nn as nn
import pytorch_lightning as pl
from entropyloss import EntropyLossEncap

class MemAE(pl.LightningModule):
    def __init__(self, backbone, opt):
        super().__init__()
        self.backbone = backbone
        self.ent_loss = EntropyLossEncap()
        self.ent_loss_weight = opt.entropy_loss_weight
        self.label = []
        self.image = []
        self.rec_losses   = []
        self.memory_items = []
        self.validation_threshold = None
        self.validation_label     = None
        self.validation_image     = None
        self.validation_att       = None

    def step(self, batch):
        x, y     = batch
        out      = self.backbone(x)
        y_hat    = out["output"]
        mem_att  = out["att"]
        rec_loss = nn.MSELoss()(y_hat, x)
        ent_loss = self.ent_loss(mem_att)
        loss     = rec_loss + self.ent_loss_weight * ent_loss
        return loss, rec_loss, ent_loss

    def forward(self, batch):   
        loss, rec_loss, ent_loss = self.step(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_reconstruction_loss', rec_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_entropy_loss', ent_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, rec_loss, ent_loss = self.step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_reconstruction_loss', rec_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_entropy_loss', ent_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # Get threshold
        x, y      = batch
        out       = self.backbone(x)
        y_hat     = out["output"]
        mem_att   = out["att"]
        r         = y_hat - x
        error_map = torch.sum(r**2, dim=1) ** 0.5
        s = error_map.size()
        error_vec   = error_map.view(s[0], -1)
        recon_error = torch.mean(error_vec, dim=-1)
        
        self.rec_losses += recon_error.cpu().tolist()
        self.label.extend(y)
        x = x.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()
        mem_att = mem_att.detach().cpu().numpy()
        self.image.extend((x[i], y_hat[i]) for i in range(len(x)))
        self.memory_items.extend(mem_att[i] for i in range(len(mem_att)))

    def on_validation_epoch_end(self):
        self.validation_threshold = self.rec_losses.copy()
        self.validation_label = self.label.copy()
        self.validation_image = self.image.copy()
        self.validation_att = self.memory_items.copy()

        self.rec_losses.clear()
        self.label.clear()
        self.image.clear()
        self.memory_items.clear()