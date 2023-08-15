import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
import matplotlib
from dataset import MNIST_Dataset
from models import *
from models.memae import AutoEncoderCov2DMem
from trainvalid import *
from visualizer import *
from utils import *
from argparse import ArgumentParser

parser = ArgumentParser(description="MemAE")

## preprocess
parser.add_argument('--data_path', default='./', type=str)
parser.add_argument('--normal_class', default='0', type=int)
parser.add_argument('--img_height', default=28, type=int)
parser.add_argument('--img_width', default=28, type=int)
parser.add_argument('--img_chn_size', default=1, type=int)
parser.add_argument('--img_crop_size', default=28, type=int)

## model - MemAE
parser.add_argument('--conv_chn_size', default=16, type=int)
parser.add_argument('--memory_dim', default=100, type=int)
parser.add_argument('--shrink_threshold', default=0.01, type=float)
parser.add_argument('--entropy_loss_weight', default=0.0002, type=float)
parser.add_argument('--device', default='mps', type=str)

## train
parser.add_argument('--num_epoch', default=1, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--seed', default=41, type=int)

opt = parser.parse_args('')

def main(opt):
    SEED = opt.seed
    set_seeds(SEED)
    submission_id = f"{parser.description}"
    print(submission_id)
    # Define datasets
    data                = MNIST_Dataset(opt=opt)
    train_set           = data.train_dataset
    valid_dataset       = data.valid_dataset
    valid_thres_dataset = data.valid_thres_dataset

    train_loader        = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    test_loader         = DataLoader(dataset=valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    test_loader_thres   = DataLoader(dataset=valid_thres_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    
    # Define model & Train
    model = MemAE(AutoEncoderCov2DMem(opt), opt)
    print(f"model info\n{model}")

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath="saved/", filename=f"{submission_id}",
            monitor="train_loss", mode="min",
            )
        ]
    print("Training ...")
    trainer = pl.Trainer(max_epochs=opt.num_epoch, accelerator="mps", devices=1, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=train_loader)

    ckpt    = torch.load(f"saved/{submission_id}.ckpt", map_location=torch.device('mps'))
    model.load_state_dict(ckpt['state_dict'])
    
    # Valid
    print("Predict ...")
    trainer.validate(model, dataloaders=test_loader_thres)
    threshold = model.validation_threshold
    #
    trainer.validate(model, dataloaders=test_loader)
    pred_rec_error = model.validation_threshold
    pred_label     = model.validation_label
    pred_image     = model.validation_image  # (x, y_hat)

    total_err = threshold + pred_rec_error
    max_error = max(total_err)
    min_error = min(total_err)
    anomaly   = [(i - min_error) / (max_error - min_error) for i in threshold]
    anomaly_threshold = max(anomaly)
    pred_rec_error    = [(i - min_error) / (max_error - min_error) for i in pred_rec_error]

    pred = [0 for _ in range(len(pred_rec_error))]
    pred_label_ano = pred_label.copy()
    for i in range(len(pred_rec_error)):
        if pred_rec_error[i] <= anomaly_threshold:
            pred_label_ano[i] = 0
        elif pred_rec_error[i] > anomaly_threshold:
            pred[i]       = 1
            pred_label_ano[i] = 1
            
    score = roc_auc_score(pred_label_ano, pred)
    # Visualization 
    plt.show(block=True)
    visual_roc_auc(pred_label_ano, pred_rec_error)
    visual_defective(threshold, pred_rec_error, pred_label_ano, anomaly_threshold)
    visual_rec_error(pred, pred_rec_error, anomaly_threshold)
    visual_normal_anomal_image(pred_label_ano, pred_image)
    visual_memory_items(model)

if __name__ == '__main__':
    main(opt)