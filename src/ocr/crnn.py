import torch

class crnn(torch.nn.Module):
    def __init__(self, img_h, n_channels, n_classes):
        super(crnn, self).__init__()
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, 64, 3, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
        )

        self.rnn_input_size = 128 * (img_h // 4)

        self.rnn = torch.nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = torch.nn.Linear(256 * 2, n_classes)

    def forward(self, x):
        features = self.cnn(x)

        b, c, h, w = features.size()

        features = features.permute(0, 3, 1, 2)
        features = features.reshape(b, w, c * h)

        rnn_out, _ = self.rnn(features)
        output = self.fc(rnn_out)

        return output
