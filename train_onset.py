
import torch 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
import torchmetrics.functional as M

from models import OnsetModel
from dataset import OnsetDataset, train_valid_split, worker_init_fn
import time

from config import config

# catch warnings 
import warnings 
warnings.filterwarnings("ignore")


# for CNNs
torch.backends.cudnn.benchmark = True

class OnsetLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, dropout=0.5, l2=1e-3, optimizer="Adam"):
        super().__init__()
        
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.dropout = dropout
        self.l2 = l2
        self.optimizer_name = optimizer

        self.example_input_array = torch.Tensor(1, config.onset.sequence_length, config.audio.n_bins, len(config.audio.n_ffts))

        self.model = OnsetModel(dropout=dropout)
        
        self.binary_accuracy = BinaryAccuracy()
        self.binary_auroc = BinaryAUROC()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        
        self.log("train/step_loss", loss)
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        
        # y_hat is already sigmoided
        acc = self.binary_accuracy(y_hat, y)
        auroc = self.binary_auroc(y_hat, y)

        self.log("train/epoch_loss", loss)
        self.log("train/epoch_acc", acc)
        self.log("train/epoch_auroc", auroc)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        
        self.log("valid/step_loss", loss)
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])

        acc = self.binary_accuracy(y_hat, y)
        auroc = self.binary_auroc(y_hat, y)    

        self.log("valid/epoch_loss", loss)
        self.log("valid/epoch_acc", acc)
        self.log("valid/epoch_auroc", auroc)

        # for hyperparameter tuning
        self.log("hp/epoch_loss", loss)

    def configure_optimizers(self):
        return getattr(torch.optim, self.optimizer_name)(self.parameters(), lr=self.learning_rate, weight_decay=self.l2)

def get_dataloaders(batch_size, num_workers=0, **kwargs):
    train_manifest, valid_manifest = train_valid_split()

    train_dataset = OnsetDataset(train_manifest)
    valid_dataset = OnsetDataset(valid_manifest)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)

    return train_loader, valid_loader

def train(train_loader, valid_loader):

    model = OnsetLightningModule(
        learning_rate=config.hyperparameters.learning_rate,
        dropout=config.hyperparameters.dropout,
        l2=config.hyperparameters.l2,
        optimizer=config.hyperparameters.optimizer
    )


    logger = TensorBoardLogger(
        save_dir=config.paths.logs,
        name="onset",
        log_graph=True
        )
    
    callbacks = []

    if config.train.checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor="valid/step_loss",
            dirpath=config.paths.checkpoints,
            filename="onset-{epoch:02d}-{val_loss:.2f}",
            save_top_k=config.train.top_k,
            mode="min",
        )

        callbacks.append(checkpoint_callback)

    if config.train.early_stopping:
        early_stopping = EarlyStopping(
            monitor="valid/step_loss",
            patience=config.train.patience,
            mode="min",
        )
    
        callbacks.append(early_stopping)

    
    trainer = pl.Trainer(
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        max_epochs=config.hyperparameters.epochs,
        logger=logger,
        callbacks=callbacks,
        auto_lr_find=config.train.find_lr
    )

    trainer.fit(model, train_loader, valid_loader)

    return trainer, model


if __name__ == "__main__":
    train_loader, valid_loader = get_dataloaders(batch_size=config.hyperparameters.batch_size, num_workers=config.train.num_workers, pin_memory=config.train.pin_memory)

    trainer, model = train(train_loader, valid_loader)