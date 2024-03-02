import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule


class SimpleMLP(LightningModule):

    def __init__(self, input_size, output_size):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.hparams['network'] = str(self.network)
        self.save_hyperparameters()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('test_loss', loss)

    def predict_step(self, batch):
        x, y = batch
        return torch.cat([self(x), y.unsqueeze(-1)], axis=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

class SimpleLSTM(LightningModule):

    def __init__(self, 
                 input_size: int, 
                 output_size: int,
                 sequence_length: int = 10,
                 lstm_hidden_size: int = 128,
                 lstm_layers: int = 1
        ):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers=lstm_layers, batch_first=True)

        self.fc_block = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

        self.hparams['lstm'] = str(self.lstm)
        self.hparams['fc_block'] = str(self.fc_block)
        self.save_hyperparameters()

    def forward(self, x):
        x, (hidden_state, cell_state) = self.lstm(x)
        x = x[:, -1, :]
        concat_features = torch.cat([x, hidden_state[-1, :, :]], dim=1)
        return self.fc_block(concat_features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('test_loss', loss)

    def predict_step(self, batch):
        x, y = batch
        return torch.cat([self(x), y.unsqueeze(-1)], axis=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    


class SimpleCNN1D(LightningModule):

    def __init__(self):
        super(SimpleCNN1D, self).__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),  
            nn.ReLU(),
            nn.Conv1d(16, 64, kernel_size=3, stride=1),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),  
            nn.Flatten(start_dim=1)  
        )

        self.fc_block = nn.Sequential(
            nn.Linear(1344, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.hparams['cnn_block'] = str(self.cnn_block)
        self.hparams['fc_block'] = str(self.fc_block)
        self.save_hyperparameters()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn_block(x)
        return self.fc_block(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('test_loss', loss)

    def predict_step(self, batch):
        x, y = batch
        return torch.cat([self(x), y.unsqueeze(-1)], axis=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    


class SimpleCNN2D(LightningModule):

    def __init__(self):
        super(SimpleCNN2D, self).__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, padding='same', padding_mode='replicate', stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1),  
            nn.Conv2d(16, 64, kernel_size=3, padding='same', stride=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3), 
            nn.Flatten(start_dim=1) 
        )

        self.fc_block = nn.Sequential(
            nn.Linear(2112, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.hparams['cnn_block'] = str(self.cnn_block)
        self.hparams['fc_block'] = str(self.fc_block)
        self.save_hyperparameters()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn_block(x)
        return self.fc_block(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1))
        self.log('test_loss', loss)

    def predict_step(self, batch):
        x, y = batch
        return torch.cat([self(x), y.unsqueeze(-1)], axis=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)