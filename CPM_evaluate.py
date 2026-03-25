import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4_mean = nn.Linear(256, 256)
        self.fc4_logvar = nn.Linear(256, 256)

        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, input_dim)

        self.fc8 = nn.Linear(256, 128)
        self.fc9 = nn.Linear(128, 64)
        self.fc10 = nn.Linear(64, 1)

        self.L2_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)

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
        L2 = self.L2_loss(x_hat, x)
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
        vae_loss = L2 + 0.001 * KLD
        prediction_loss = self.L2_loss(target_pred, target_real)
        return vae_loss + prediction_loss


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def eval_model(model, data, target_index):
    model.eval()

    # 准备数据
    if isinstance(data, pd.DataFrame):
        data_values = data.values
    else:
        data_values = data

    device = next(model.parameters()).device

    all_data = torch.tensor(data_values, dtype=torch.float32).to(device)

    num_samples, num_features = all_data.shape
    feature_indices = [i for i in range(num_features) if i != target_index]

    losses = []

    # 1. Calculate the loss on the original unmodified data
    with torch.no_grad():
        x = all_data[:, feature_indices]
        target_real = all_data[:, target_index].view(-1, 1)

        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        target_pred = model.predict_target(z)

        original_mse = nn.MSELoss(reduction='mean')(target_pred, target_real).item()
        losses.append(original_mse)

    # 2. Calculate the loss after zeroing out each feature iteratively (Counterfactual Perturbation)
    for i, feature_idx in enumerate(feature_indices):
        with torch.no_grad():
            x_modified = x.clone()
            x_modified[:, i] = 0

            mean, logvar = model.encode(x_modified)
            z = model.reparameterize(mean, logvar)
            target_pred = model.predict_target(z)

            mse = nn.MSELoss(reduction='mean')(target_pred, target_real).item()
            losses.append(mse)

    return np.array(losses)


csv_file = './data/sim1_test.csv'
data = pd.read_csv(csv_file)
print('data:\n', data)


scaler = StandardScaler()

data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
print('data:\n', data)

variable_num = len(data.columns)
results = []

for target_index in range(variable_num):
    target_name = data.columns[target_index]

    model = VAE(input_dim=variable_num - 1).to(device)

    model.load_state_dict(torch.load(f'./model/model_{target_name}/model_epoch_100.pt',weights_only=True))

    losses = eval_model(model, data, target_index)

    loss_names = ["Original"] + [f"{col}_zeroed" for col in data.columns if col != target_name]
    result_df = pd.DataFrame({target_name: losses}, index=loss_names).T
    results.append(result_df)
    print('------------------------------')
    print('result_df:\n', result_df)

    os.makedirs('./result', exist_ok=True)
    result_df.to_csv(f'./result/losses_target_{target_name}_epoch=100.csv')

# 合并所有结果并保存到一个CSV文件
all_results_df = pd.concat(results)
all_results_df.to_csv('./result/all_losses_epoch=100.csv')