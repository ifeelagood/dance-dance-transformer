
import torch 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torchmetrics.functional as M

from models import OnsetModel
from dataset import OnsetDataset, train_valid_split, worker_init_fn
import time

# catch warnings 
import warnings 
warnings.filterwarnings("ignore")


# for CNNs
torch.backends.cudnn.benchmark = True

class OnsetLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=256, num_workers=0, pin_memory=True):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
        self.model = OnsetModel()
        
        self.accuracy = lambda y_hat, y: torch.mean((y_hat > 0.5) == y)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        # y_hat is already sigmoided
        acc = self.accuracy(y_hat, y)
        
        self.log("valid_loss", loss)
        self.log("valid_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train(model, train_loader, valid_loader):
    logger = TensorBoardLogger("lightning_logs", name="onset")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath="lightning_logs/onset/checkpoints",
        filename="onset-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        mode="min",
    )
    
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        patience=10,
        mode="min",
    )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        auto_lr_find=True,
    )

    trainer.fit(model, train_loader, valid_loader)


def overfit_batch(model, batch):
    logger = TensorBoardLogger("lightning_logs", name="onset_sanity")
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1000,
        logger=logger,
        overfit_batches=16,
    )

    trainer.fit(model, batch)

if __name__ == "__main__":
    train_manifest, valid_manifest = train_valid_split()

    train_dataset = OnsetDataset(train_manifest)
    valid_dataset = OnsetDataset(valid_manifest)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, num_workers=0)
    
    model = OnsetLightningModule(learning_rate=1e-3)
    
    train(model, train_loader, valid_loader) 