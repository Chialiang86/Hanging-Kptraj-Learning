import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        latent = self.fc(hidden.squeeze())
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, output_size, batch_first=True)

    def forward(self, x, hidden, cell):
        x = self.fc(x)
        x = x.unsqueeze(0)
        x, (hidden, cell) = self.lstm(x, (hidden, cell))
        return x, hidden, cell

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)

    def forward(self, x):
        latent = self.encoder(x)
        hidden = latent.unsqueeze(0)
        cell = torch.zeros_like(hidden)
        reconstructed_x, _, _ = self.decoder(latent, hidden, cell)
        return reconstructed_x, latent

def reconstruction_loss(reconstructed_x, x):
    reconstruction_loss = nn.MSELoss()(reconstructed_x, x)
    return reconstruction_loss

if __name__=="__main__":
    # define the device and the input size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 3

    # define the model, loss function, and optimizer
    model = VAE(input_size, 128, 32).to(device)
    criterion = reconstruction_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # create a synthetic dataset of 3D positions
    num_samples = 1000
    sequence_length = 50
    positions = torch.randn(num_samples, sequence_length, input_size).to(device)

    # define a dataloader
    dataloader = DataLoader(positions, batch_size=64, shuffle=True)

    # training loop
    for epoch in range(100):
        running_loss = 0
        for data in dataloader:
            positions = data
            optimizer.zero_grad()
            reconstructed_positions, latent = model(positions)
            loss = criterion(reconstructed_positions, positions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss = {running_loss/len(dataloader):.4f}')