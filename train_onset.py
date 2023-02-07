
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
import torchmetrics.functional as M


from dataset import OnsetDataset, train_valid_split, worker_init_fn
import time

from config import config

# catch warnings 
import warnings 
warnings.filterwarnings("ignore")


# for CNNs
torch.backends.cudnn.benchmark = True

class OnsetLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, dropout=0.5, l2=1e-3, optimizer="Adam", hidden_size=100, num_layers=2, bidirectional=True):
        super().__init__()
        
        self.save_hyperparameters()

        # parameters
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.l2 = l2
        self.optimizer_name = optimizer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # for graphing forward pass
        self.example_input_array = torch.Tensor(1, config.onset.sequence_length, config.audio.n_bins, len(config.audio.n_ffts))

        # layers
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(7, 3))
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3))
        self.pool = torch.nn.MaxPool2d(kernel_size=(1, 3))

        self.lstm = torch.nn.LSTM(
            input_size=160,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )


        self.fc1 = torch.nn.Linear(200, 100)
        self.fc2 = torch.nn.Linear(100, 1)

        self.binary_accuracy = BinaryAccuracy()
        self.binary_auroc = BinaryAUROC()

        self.sigmoid = torch.nn.Sigmoid()

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        )

    def forward(self, x):
        # permute to (B, C, T, F)
        x = x.permute(0, 3, 1, 2) 
    
        # conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # permute to (B, T, F, C) for LSTM
        x = x.permute(0, 2, 3, 1)

    
        # flatten to 
        x = x.reshape(x.size(0), x.size(1), -1)

        # lstm
        hidden = self.init_hidden(x.size(0), x.device)

        x, hidden = self.lstm(x, hidden)

        # get last frame for FC layers
        x = x[:, -1, :]

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # sigmoid
        x = self.sigmoid(x)

        return x.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        
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
        loss = F.binary_cross_entropy(y_hat, y)
        
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

def find_lr(train_loader, valid_loader):
    model = OnsetLightningModule(
        learning_rate=config.hyperparameters.learning_rate,
        dropout=config.hyperparameters.dropout,
        l2=config.hyperparameters.l2,
        optimizer=config.hyperparameters.optimizer
    )

    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        callbacks=[lr_logger],
        accelerator=config.train.accelerator,
        devices=config.train.devices,
    )

    trainer.fit(model, train_loader, valid_loader)

    return lr_logger.lr_scheduler

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
            monitor="valid/epoch_loss",
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