import argparse
import json

import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import wandb

from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
import torchmetrics.functional as M

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import matplotlib.pyplot as plt

import webdataset as wds

from dataset import get_dataset, get_dataloader
from config import config
from models.onset import LSTM_A, CNN_A, Classifier

# catch warnings 
import warnings 
warnings.filterwarnings("ignore")

# for CNNs
torch.backends.cudnn.benchmark = True

class OnsetLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, weight_decay=0, momentum=0, dropout=0.5, optimizer="Adam", hidden_size=200, num_layers=2, bidirectional=True):
        super().__init__()
        
        self.save_hyperparameters()

        # parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dropout = dropout
        self.optimizer_name = optimizer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # for graphing forward pass
        self.example_input_array = torch.Tensor(1, config.onset.context_radius * 2 + 1, config.audio.n_bins, len(config.audio.n_ffts))

        # layers
        self.cnn = CNN_A(in_channels=len(config.audio.n_ffts))
    
        self.lstm = LSTM_A(input_size=165, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    
        self.classifier = Classifier(input_size=200, output_size=1, return_logits=False)
    

        self.binary_accuracy = BinaryAccuracy()
        self.binary_precision = BinaryPrecision()
        self.binary_recall = BinaryRecall()
        self.binary_auroc = BinaryAUROC()
        self.binary_f1_score = BinaryF1Score()

    def init_hidden(self, batch_size, device):
        # layer initalisation for binary class imbalance
        num_directions = 2 if self.bidirectional else 1

        h0 = torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_size, device=device)

        return (h0, c0)
        
    def forward(self, x, difficulty=None):
        # permute to (B, C, T, F)
        x = x.permute(0, 3, 1, 2) 
    
        # CNN
        x = self.cnn(x)

        # permute to (B, T, F, C), then flatten to (B, T, F*C)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), x.size(1), -1)
        

        # add onehot difficulty as a feature to each frame
        if difficulty is not None:
            difficulty = difficulty.unsqueeze(1).repeat(1, x.size(1), 1)
            x = torch.cat([x, difficulty], dim=2)
        else:
            x = torch.cat([x, torch.zeros(x.size(0), x.size(1), 5, device=x.device)], dim=2)

        # lstm
        x = self.lstm(x)
    
        # get last frame for FC layers
        x = x[:, -1, :]

        # FC layers
        x = self.classifier(x).squeeze(1)
    
        return x

    def training_step(self, batch, batch_idx):
        features, difficulty, y = batch
        y_hat = self.forward(features, difficulty)
        loss = F.binary_cross_entropy(y_hat, y)
        

        # logging
        self.log("train/loss", loss)
        self.log("train/accuracy",  self.binary_accuracy(y_hat, y))
        self.log("train/precision", self.binary_precision(y_hat, y))
        self.log("train/recall", self.binary_recall(y_hat, y))
        self.log("train/f1", self.binary_f1_score(y_hat, y))
        
        return loss


    def validation_step(self, batch, batch_idx):
        features, difficulty, y = batch
        y_hat = self.forward(features, difficulty)
        loss = F.binary_cross_entropy(y_hat, y)
            
        # logging
        self.log("valid/loss", loss)
        self.log("valid/accuracy",  self.binary_accuracy(y_hat, y))
        self.log("valid/precision", self.binary_precision(y_hat, y))
        self.log("valid/recall", self.binary_recall(y_hat, y))
        self.log("valid/f1", self.binary_f1_score(y_hat, y))
        
    
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])

        # logging
        self.log("valid/epoch_loss", loss)
        self.log("valid/epoch_accuracy",  self.binary_accuracy(y_hat, y))
        self.log("valid/epoch_f1", self.binary_f1_score(y_hat, y))

    def configure_optimizers(self):
        kwargs = {"lr": self.learning_rate, "weight_decay": self.weight_decay}

        # add momentum if optimizer is SGD, RMSprop or 
        if self.optimizer_name in ["SGD", "RMSprop"]:
            kwargs["momentum"] = self.momentum
            
        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(), **kwargs)
        
        return optimizer

def train(args, train_loader, valid_loader):
    # init wandb
    wandb.init(project="onset", entity="ifag")
    
    # log important hyperparameters that are not stored in lightning module
    wandb.config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "gradient_clip": args.gradient_clip,
        "sample_rate": config.audio.sample_rate,
        "n_ffts": config.audio.n_ffts,
        "hop_length": config.audio.hop_length,
        "n_bins": config.audio.n_bins,
        "log_scale": config.audio.log,
        "normalize": config.audio.normalize,
        "context_radius": config.onset.context_radius,
    })
    
    # define model
    model = OnsetLightningModule(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        dropout=args.dropout,
        optimizer=args.optimizer,
        hidden_size=200,
        num_layers=2,
        bidirectional=False,
    )
    
    # define logger
    logger = WandbLogger(project="onset", entity="ifag", log_model=True)
    
    callbacks = []

    if config.callbacks.lr_monitor:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        

    if args.checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor=config.callbacks.checkpoint.monitor,
            dirpath=config.paths.checkpoints,
            save_top_k=args.top_k,
            mode=config.callbacks.checkpoint.mode,
        )

        callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor=config.callbacks.early_stopping.monitor,
            patience=config.callbacks.early_stopping.patience,
            mode=config.callbacks.early_stopping.mode
        )
    
        callbacks.append(early_stopping)

    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=4000,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=100
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # save best model
    if args.checkpoint:
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f"Best model path: {best_model_path}")
        
    return trainer, model

def find_lr(args, train_loader, valid_loader):

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=config.device.strategy,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=100,
        max_epochs=1,
        max_steps=10000
    )
    
    model = OnsetLightningModule(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        dropout=args.dropout,
        optimizer=args.optimizer,
        hidden_size=200,
        num_layers=2,
        bidirectional=False,
    )

    lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=valid_loader, num_training=10000, min_lr=1e-6, max_lr=1e-1)
    fig = lr_finder.plot(suggest=True)
    plt.show()

    fig.show()
    fig.savefig("lr_finder.png")

    print(f"suggested learning rate: {lr_finder.suggestion()}")

    return lr_finder

def objective(trial):
    wandb.init(project="onset", entity="ifag")
    
    # get trial hyperparameters
    hp = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1.0),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1.0),
        "momentum": trial.suggest_uniform("momentum", 0.0, 1.0),
        "dropout": trial.suggest_uniform("dropout", 0.0, 1.0),
        "optimizer": trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"]),
#        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "gradient_clip": trial.suggest_uniform("gradient_clip", 0.0, 10.0),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    }

    # add batch size to wandb config
    wandb.config.update({
        "epochs": args.epochs,
        "batch_size": hp["batch_size"],
        "gradient_clip": hp["gradient_clip"],
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "sample_rate": config.audio.sample_rate,
        "n_ffts": config.audio.n_ffts,
        "hop_length": config.audio.hop_length,
        "n_bins": config.audio.n_bins,
        "log_scale": config.audio.log,
        "normalize": config.audio.normalize,
        "context_radius": config.onset.context_radius,
    })

    
    model = OnsetLightningModule(
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        momentum=hp["momentum"],
        dropout=hp["dropout"],
        optimizer=hp["optimizer"],
        bidirectional=False,
        hidden_size=200,
        num_layers=2,
    )

    train_loader = get_dataloader(get_dataset("train"), batch_size=hp["batch_size"], batched_dataloder=True, pin_memory=config.dataloader.pin_memory, num_workers=config.dataloader.num_workers)
    valid_loader = get_dataloader(get_dataset("valid"), batch_size=hp["batch_size"], batched_dataloder=True, pin_memory=config.dataloader.pin_memory, num_workers=config.dataloader.num_workers)

    logger = WandbLogger(name="tune", project="onset", entity="ifag", log_model=True)
 
    trainer = pl.Trainer(
        accelerator=config.device.accelerator,
        devices=config.device.devices,
        strategy=config.device.strategy,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=config.tuning.pruning.monitor)],
        val_check_interval=4000,
        gradient_clip_val=hp["gradient_clip"],
        accumulate_grad_batches=config.hyperparameters.accumulate_grad_batches,
        log_every_n_steps=100,
    )
        
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    return trainer.callback_metrics[config.tuning.monitor]

 
def get_args():
    parser = argparse.ArgumentParser(description="Train the onset detection model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # README - ArgumentsDefaultsHelpFormatter requires all arguments use a help string, even if it is empty

    # train/optimise
    parser.add_argument("action", type=str, choices=["train", "tune", "find_lr"], help="Whether to train or tune the model.")


    # create groups
    p_hp = parser.add_argument_group("hyperparameters")
    p_dataloader = parser.add_argument_group("dataloader")
    p_callbacks = parser.add_argument_group("callbacks")
    p_device = parser.add_argument_group("device")
    p_tuning = parser.add_argument_group("tuning")
    
    empty = " - "
    
    # hyperparameters
    p_hp.add_argument("--epochs", type=int, default=config.hyperparameters.epochs, help=empty)
    p_hp.add_argument("--batch-size", type=int, default=config.hyperparameters.batch_size, help=empty)
    p_hp.add_argument("--learning-rate", type=float, default=config.hyperparameters.learning_rate, help=empty)
    p_hp.add_argument("--weight-decay", type=float, default=config.hyperparameters.weight_decay, help=empty)
    p_hp.add_argument("--momentum", type=float, default=config.hyperparameters.momentum, help=empty)
    p_hp.add_argument("--dropout", type=float, default=config.hyperparameters.dropout, help=empty)
    p_hp.add_argument("--optimizer", type=str, default=config.hyperparameters.optimizer, help=empty)
    p_hp.add_argument("--gradient-clip", type=float, default=config.hyperparameters.gradient_clip, help=empty)
    p_hp.add_argument("--accumulate-grad-batches", type=int, default=config.hyperparameters.accumulate_grad_batches, help=empty)

    # dataloader
    p_dataloader.add_argument("--num-workers", type=int, default=config.dataloader.num_workers, help=empty)
    p_dataloader.add_argument("--pin-memory", type=bool, default=config.dataloader.pin_memory, help=empty)
    
    # device
    p_device.add_argument("--accelerator", type=str, default=config.device.accelerator, help=empty)
    p_device.add_argument("--devices", type=int, default=config.device.devices, help=empty)
    p_device.add_argument("--strategy", type=int, default=config.device.strategy, help=empty)
    
    # callbacks
    p_callbacks.add_argument("--checkpoint", type=bool, default=config.callbacks.checkpoint.enable, help=empty)
    p_callbacks.add_argument("--early-stopping", type=bool, default=config.callbacks.early_stopping.enable, help=empty)
    p_callbacks.add_argument("--top-k", type=int, default=config.callbacks.checkpoint.top_k, help=empty)
    p_callbacks.add_argument("--patience", type=int, default=config.callbacks.early_stopping.patience, help=empty)
    
    # tuning
    p_tuning.add_argument("--n-trials", type=int, default=config.tuning.n_trials, help=empty)
    p_tuning.add_argument("--n-jobs", type=int, default=config.tuning.n_jobs, help=empty)
    p_tuning.add_argument("--timeout", type=int, default=config.tuning.timeout, help=empty)
    p_tuning.add_argument("--max-epochs", type=int, default=config.tuning.max_epochs, help=empty)
    p_tuning.add_argument("--direction", type=str, default=config.tuning.direction, help=empty)
    p_tuning.add_argument("--prune", type=bool, default=config.tuning.pruning.enable, help="Enable early pruning of unpromising trials.")
    p_tuning.add_argument("--pruning-monitor", type=str, default=config.tuning.pruning.monitor, help=empty)
    p_tuning.add_argument("--monitor", type=str, default=config.tuning.monitor, help=empty)


    return parser.parse_args()

def get_dataloaders(args, batched_dataloader=False):

    train_dataset = get_dataset("train")
    valid_dataset = get_dataset("valid")

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, batched_dataloder=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    valid_loader = get_dataloader(valid_dataset, batch_size=args.batch_size, batched_dataloder=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    return train_loader, valid_loader


if __name__ == "__main__":
    args = get_args()

    if args.action == "train":
        train_loader, valid_loader = get_dataloaders(args, batched_dataloader=False)

        trainer, model = train(args, train_loader, valid_loader)

    elif args.action == "tune":

        pruner = optuna.pruners.MedianPruner() if args.prune else None
    
        study = optuna.create_study(
            direction=args.direction,
            pruner=pruner,
            study_name="onset_detection"
        )
        
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout
        )
        
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        # save the best trial
        with open("best_trial.json", "w") as f:
            json.dump(trial.params, f, indent=4)
        
    elif args.action == "find_lr":
        train_loader, valid_loader = get_dataloaders(args, batched_dataloader=False)
        find_lr(args, train_loader, valid_loader)
