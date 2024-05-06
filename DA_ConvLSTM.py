import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention1D, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd to maintain the sequence length")
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MineralCNNLSTM_Flatten(nn.Module):
    def __init__(self, num_classes, hidden_size=300, num_layers=1):
        super(MineralCNNLSTM_Flatten, self).__init__()

        # Existing CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=21, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Channel attention layer after the last CNN layer
        self.ca = ChannelAttention(num_channels=64)
        self.sa1 = SpatialAttention1D(kernel_size=5)
        self.sa2 = SpatialAttention1D(kernel_size=5)
        self.sa3 = SpatialAttention1D(kernel_size=5)

        # Calculate the total number of features after the last conv layer
        self.total_features = self._get_conv_output((1, 1, 950))

        # LSTM layer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.total_features, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=0.2)

        # Define fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2048, num_classes)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2048, 1)
        )

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(shape)
            output = self.conv1(input)
            output = self.conv2(output)
            output = self.conv3(output)
            return output.numel()

    def forward(self, x):
        # CNN layers
        x = self.conv1(x)
        x = self.sa1(x) * x
        x = self.conv2(x)
        x = self.sa2(x) * x
        x = self.conv3(x)
        x = self.sa3(x) * x

        # Apply channel attention
        x = self.ca(x) * x

        # Flatten the output and then unsqueeze to add a time step dimension
        x = x.view(x.size(0), -1).unsqueeze(1)

        # LSTM layer
        x, _ = self.lstm(x)

        # Take the output of the last time step
        out = x[:, -1, :]

        # Fully connected layer
        out_1 = self.fc1(out)
        out_2 = self.fc2(out)

        return out_1, out_2
