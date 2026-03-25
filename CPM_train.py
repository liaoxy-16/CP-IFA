import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()

        # Encoder architecture
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4_mean = nn.Linear(256, 256)
        self.fc4_logvar = nn.Linear(256, 256)

        # Decoder architecture
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, input_dim)

        # Target prediction module
        self.fc8 = nn.Linear(256, 128)
        self.fc9 = nn.Linear(128, 64)
        self.fc10 = nn.Linear(64, 1)

        self.L2_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc4_mean(x)
        logvar = self.fc4_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        x = F.relu(self.fc5(z))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar, z

    def predict_target(self, z):
        target = F.relu(self.fc8(z))
        target = F.relu(self.fc9(target))
        target = self.fc10(target)
        return target

    def loss_function(self, x, x_hat, mean, logvar, target_real, target_pred):
        # Reconstruction Loss
        L2 = self.L2_loss(x_hat, x)
        # KL Divergence
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
        vae_loss = L2 + 0.001 * KLD
        # Prediction Loss
        prediction_loss = self.L2_loss(target_pred, target_real)
        return vae_loss + prediction_loss

    def train_model(self, data, epochs, batch_size, target_index, target_name):
        feature_indices = [i for i in range(data.shape[1]) if i != target_index]

        current_device = next(self.parameters()).device

        for epoch in range(epochs):
            total_loss = 0
            num_samples = len(data)

            # Shuffle data for each epoch
            indices = np.random.permutation(num_samples)
            data_shuffled = data[indices]

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                batch_data = data_shuffled[i:i + batch_size]

                batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(current_device)

                x = batch_tensor[:, feature_indices]
                target_real = batch_tensor[:, target_index].reshape(-1, 1)

                # Forward pass
                x_hat, mean, logvar, z = self(x)
                target_pred = self.predict_target(z)

                # Compute loss
                loss = self.loss_function(x, x_hat, mean, logvar, target_real, target_pred)
                total_loss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}: average loss = {total_loss / num_samples}")

            if epoch % 50 == 0:
                save_model(self, f"./model/model_{target_name}/model_epoch_{epoch}.pt")


def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)


# Load dataset
df = pd.read_csv('./data/sim1_test.csv')
print('df:\n', df)

# Standardize data using Z-Score
scaler = StandardScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

data = df.values

num_samples, input_dim = data.shape
column_names = df.columns

batch_size = 32
for target_index in range(input_dim):
    target_name = column_names[target_index]
    print(f"Training model with target variable: {target_name}")


    model = VAE(input_dim - 1).to(device)
    model.train_model(data, epochs=201, batch_size=batch_size, target_index=target_index, target_name=target_name)