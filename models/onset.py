import torch

class OnsetCNN(torch.nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # 10 filter kernels that are 7-wide in time and 3-wide in frequency
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=(7, 3))
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3))
        self.pool = torch.nn.MaxPool2d(kernel_size=(1,3))

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # layer 1
        x = self.pool(self.relu(self.conv2(x))) # layer 2

        return x # 7, 8 * 20

class OnsetLSTM(torch.nn.Module):
    def __init__(self, n_features=160, num_layers=2, hidden_size=128):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size):
        # (num_layers, batch_size, hidden_size)
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x, hidden):
        out, _ = self.lstm(x, hidden)
        out = self.fc(out)

        return out

class OnsetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    

        self.cnn = OnsetCNN()
        self.lstm = OnsetLSTM()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(7, 1),
        )

        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]

        # permute to channels first
        x = x.permute(0, 2, 3, 1) # (B, T, F, C)
    

        # CNN
        out = self.cnn(x)
        print("cnn out shape: ", out.shape)

        # permute to (B, T, F, C), flatten channels and features
        out = out.permute(0, 2, 3, 1)
        out = out.view(batch_size, -1, 160)

        # LSTM
        hidden = self.lstm.init_hidden(batch_size)
        out = self.lstm(out, hidden)
        print("lstm out shape: ", out.shape)

        # squeeze out the last dimension
        out = out.squeeze(-1)

        # FC
        out = self.fc(out)
        print("fc out shape: ", out.shape)

        # sigmoid activation
        out = self.activation(out)

        return out

if __name__ == "__main__":
    x = torch.randn(32, 15, 80, 3)

    # permute to channels second
    x = x.permute(0, 3, 1, 2) # (B, C, T, F)

    y_hat = OnsetModel()(x)

    print(y_hat.shape)