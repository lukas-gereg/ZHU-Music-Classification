import torch
from torch import nn

from models.base_model import BaseModel


class OneDCnnRnnMusicModel(BaseModel):
    def __init__(self, default_params: dict):
        super().__init__(default_params)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

        inputs_list = [default_params["color_channels"]] + default_params["cnn_sizes"]

        cnns: list[nn.Module] = [
            nn.Conv2d(in_channels=value, out_channels=inputs_list[idx + 1], kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
            for idx, value in enumerate(inputs_list[:-1])]

        self.batchnorm = nn.BatchNorm2d(num_features=cnns[0].out_channels)

        self.cnn = self.create_cnn(cnns)

        self.gru = nn.GRU(input_size=self.calculate_gru_input_size(default_params['image_size'], default_params['color_channels']),
                          hidden_size=default_params['hidden_size'], num_layers=default_params['num_layers'], batch_first=True)

        fc_inputs_list = [self.gru.hidden_size] + default_params['fc_size']

        self.fc = nn.Sequential(
            *[nn.Sequential(nn.Linear(in_features=value, out_features=fc_inputs_list[idx + 1]), self.relu)
              for idx, value in enumerate(fc_inputs_list[:-1])],
            nn.Sequential(nn.Linear(in_features=default_params['fc_size'][-1], out_features=default_params['num_classes']), nn.Softmax(dim=1))
        )

    def calculate_gru_input_size(self, image_size, color_channels):
        image_size_x = image_size[0] if isinstance(image_size, tuple) else image_size
        image_size_y = image_size[1] if isinstance(image_size, tuple) else image_size

        x = self.cnn(torch.rand(1, color_channels, image_size_x, image_size_y))

        batch_size, channels, height, sequence_length = x.shape

        return channels * height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)

        batch_size, channels, height, sequence_length = x.shape

        x = x.view(batch_size, channels * height, sequence_length)
        x = x.permute(0, 2, 1)

        x, _ = self.gru(x)

        x = self.fc(x[:, -1, :])

        return x

    def create_cnn(self, cnns):
        cnn = nn.Sequential()

        cnn.append(nn.Sequential(nn.Sequential(cnns[0], self.relu), self.batchnorm, self.maxpool))

        for cnn_layer in cnns[1:]:
            cnn.append(nn.Sequential(nn.Sequential(cnn_layer, self.relu), self.dropout, self.maxpool))

        return cnn
