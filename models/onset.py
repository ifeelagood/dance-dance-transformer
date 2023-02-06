import torch
import torch.nn.functional as F

class OnsetCNN_Head(torch.nn.Module):
    """
        Front-End from https://arxiv.org/pdf/1806.06773.pdf

        conv    3x7     10x
        maxpool 3x1
        conv    3x3     20x
        maxpool 3x1
        dropout
    """
    def __init__(self, in_channels=3, dropout=0.5):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=(3, 7))
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3))
        self.pool = torch.nn.MaxPool2d(kernel_size=(1,3))
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # 10x9x26 -> 20x7x24 -> 20x7x8

        # dropout 
        x = self.dropout(x)

        return x

class OnsetCNN_Tail(torch.nn.Module):
    """
    Back-End D from https://arxiv.org/pdf/1806.06773.pdf

    conv    3x3     60x
    conv    3x3     60x
    conv    3x3     60x
    flatten
    dropout
    """

    def __init__(self, in_channels=20, dropout=0.5):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=60, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=60, out_channels=60, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(in_channels=60, out_channels=60, kernel_size=3)

        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=3)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)
        x = self.dropout(x)

        return x




class OnsetModel(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
    
        self.cnn_head = OnsetCNN_Head(dropout=dropout)
        self.cnn_tail = OnsetCNN_Tail(dropout=dropout)
    
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(300, 128),
            torch.nn.Linear(128, 1)
        )

        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        # permute to (B, C, T, F)
        x = x.permute(0, 3, 1, 2)

        # CNN
        out = self.cnn_tail(self.cnn_head(x))

        # FC + sigmoid
        out = self.activation(self.fc(out))

        # squeeze channel dimension
        out = out.squeeze(1)

        return out

if __name__ == "__main__":
    # sanity check
    x = torch.randn(32, 15, 80, 3)
    y_hat = OnsetModel()(x)

    print(y_hat.shape)