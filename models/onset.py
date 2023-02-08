import torch
import torch.nn.functional as F

INPUT_SHAPE = (3, 15, 80)

class CNN_A(torch.nn.Module):
    """
        CNN block for onset detection, as originally described in: 
        "Improved musical onset detection with convolutional neural networks", Schlüter & Böck (2014) 
    
        Input shape: (batch_size, 3, 15, 80)
        Output shape: (batch_size, 20, 7, 8)
    """
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=(7, 3))
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3))
    
        self.pool = torch.nn.MaxPool2d(kernel_size=(1, 3))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
    
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

class LSTM_A(torch.nn.Module):
    """
        LSTM block for onset detection, as originally described in:
        "Dance Dance Convolution", 2017
    
    """

    def __init__(self, input_size=160, hidden_size=100, num_layers=2, dropout=0.5, bidirectional=False):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
        
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        
    def init_hidden(self, batch_size, device):
        directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * directions, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * directions, batch_size, self.hidden_size, device=device)
        return (h0, c0)
        
    def forward(self, x):
        hidden = self.init_hidden(x.shape[0], x.device)
        
        x, _ = self.lstm(x, hidden)
        
        return x

class Classifier(torch.nn.Module):
    """
        Classifier block for onset detection, as originally described in:
        "Dance Dance Convolution", 2017
    
    """

    def __init__(self, input_size=100, output_size=1, return_logits=False):
        super().__init__()
        
        self.return_logits = return_logits
        
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        if not self.return_logits:
            x = torch.sigmoid(x)

        return x
    
if __name__ == '__main__':
    cnn = CNN_A(in_channels=3)
    lstm = LSTM_A(input_size=160, hidden_size=100, num_layers=2, dropout=0.5, bidirectional=False)
    classifier = Classifier(input_size=100, output_size=1, return_logits=False)

    x = torch.randn(1, *INPUT_SHAPE)
    out = cnn(x)

    print(f"out.shape: {out.shape}")

    # transpose and flatten, preserving width dimension
    out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], out.shape[2], -1)
    print(f"out.shape: {out.shape}")
    
    out = lstm(out)
    print(f"out.shape: {out.shape}")

    out = classifier(out)    
