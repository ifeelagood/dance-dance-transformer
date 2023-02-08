
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
import torchmetrics.functional as M
import matplotlib.pyplot as plt

from dataset import OnsetDataset, train_valid_split, worker_init_fn
import time
import argparse



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
        
        # compute metrics
        precision = self.binary_precision(y_hat, y)
        recall = self.binary_recall(y_hat, y)    
    
        # logging
        self.log("train/step_loss", loss)
        self.log("train/step_precision", precision)
        self.log("train/step_recall", recall)
        
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        
        # epoch metrics
        acc = self.binary_accuracy(y_hat, y)
        binary_f1_score = self.binary_f1_score(y_hat, y)
        binary_precision = self.binary_precision(y_hat, y)
        binary_recall = self.binary_recall(y_hat, y)

        # logging
        self.log("train/epoch_loss", loss)
        self.log("train/epoch_acc", acc)
        self.log("train/epoch_f1_score", binary_f1_score)
        self.log("train/epoch_precision", binary_precision)
        self.log("train/epoch_recall", binary_recall)
    

    def validation_step(self, batch, batch_idx):
        features, difficulty, y = batch
        y_hat = self.forward(features, difficulty)
        loss = F.binary_cross_entropy(y_hat, y)
            
        # compute metrics
        precision = self.binary_precision(y_hat, y)
        recall = self.binary_recall(y_hat, y)
        
        # logging
        self.log("valid/step_loss", loss)
        self.log("valid/step_precision", precision)
        self.log("valid/step_recall", recall)
        
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])

        # compute metrics
        acc = self.binary_accuracy(y_hat, y)
        f1_score = self.binary_f1_score(y_hat, y)
        precision = self.binary_precision(y_hat, y)
        recall = self.binary_recall(y_hat, y)
    
        # logging
        self.log("valid/epoch_loss", loss)
        self.log("valid/epoch_acc", acc)
        self.log("valid/epoch_f1_score", f1_score)
        self.log("valid/epoch_precision", precision)
        self.log("valid/epoch_recall", recall)


    def configure_optimizers(self):
        kwargs = {"lr": self.learning_rate, "weight_decay": self.weight_decay}

        # add momentum if optimizer is SGD, RMSprop or 
        if self.optimizer_name in ["SGD", "RMSprop"]:
            kwargs["momentum"] = self.momentum
            
        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(), **kwargs)
        
        return optimizer

def get_dataloaders(batch_size, num_workers=0, **kwargs):
    train_manifest, valid_manifest = train_valid_split()

    train_dataset = OnsetDataset(train_manifest)
    valid_dataset = OnsetDataset(valid_manifest)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)

    return train_loader, valid_loader


def train(args, train_loader, valid_loader):
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


    logger = TensorBoardLogger(
        save_dir=config.paths.logs,
        name="onset",
        log_graph=True
    )
    
    
    callbacks = []

    if config.callbacks.lr_monitor:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        

    if args.checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor=config.callbacks.checkpoint.monitor,
            dirpath=config.paths.checkpoints,
            filename="onset-{epoch:02d}-{val_loss:.5f}",
            save_top_k=args.top_k,
            mode="min",
        )

        callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor=config.callbacks.early_stopping.monitor,
            patience=config.callbacks.early_stopping.patience,
            mode="min",
        )
    
        callbacks.append(early_stopping)

    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        limit_train_batches=10000,
        val_check_interval=4000,
        gradient_clip_val=args.gradient_clip,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # save best model
    if args.checkpoint:
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f"Best model path: {best_model_path}")
        
    

    return trainer, model

def get_args():
    parser = argparse.ArgumentParser(description="Train the onset detection model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # README - ArgumentsDefaultsHelpFormatter requires all arguments use a help string, even if it is empty

    # create groups
    p_hp = parser.add_argument_group("hyperparameters")
    p_dataloader = parser.add_argument_group("dataloader")
    p_callbacks = parser.add_argument_group("callbacks")
    p_device = parser.add_argument_group("device")
    
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
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()


    train_loader, valid_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    
    trainer, model = train(args, train_loader, valid_loader)
