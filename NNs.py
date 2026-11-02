import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from engine_sim import *

sim = L15A1_sim()

torch.manual_seed(42)
np.random.seed(42)

initialization = sim.generate_dataset(sample=25000)

X_initialization = initialization[["RPM", "MAP_mbar", "Target_AFR", "Intake_Temp", "Hondata_Raw_Fuel"]].to_numpy()
Ys_initialization = initialization[["Ignition_Advance_Deg"]].to_numpy().reshape(-1,1)
Yf_initialization = (initialization["Simulated_AFR"] / initialization["Target_AFR"]).to_numpy().reshape(-1,1)

scaler_X = StandardScaler().fit(X_initialization)
scaler_Ys = StandardScaler().fit(Ys_initialization)
scaler_Yf = StandardScaler().fit(Yf_initialization)



class SimEngineData(Dataset):
    def __init__(self,sample_count):
        df = sim.generate_dataset(sample=sample_count)
        X_data = df[["RPM", "MAP_mbar", "Target_AFR", "Intake_temp", "Hondata_Raw_Fuel"]].to_numpy()
        Ys_data = df["Ignition_Advance_Deg"].to_numpy().reshape(-1, 1)
        Yf_data = (df["Simulated_AFR"] / df["Target_AFR"]).to_numpy().reshape(-1, 1)

        self.X = torch.tensor(scaler_X.transform(X_data), dtype=torch.float32)
        self.Ys = torch.tensor(scaler_Ys.transform(Ys_data), dtype=torch.float32)
        self.Yf = torch.tensor(scaler_Yf.transform(Yf_data), dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Ys[idx], self.Yf[idx]



validation_data = SimEngineData(10000)
validation_loader = DataLoader(validation_data, batch_size=128, shuffle=False)


class IgnitionTimingMLP(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class FuelRatioMLP(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

IgnitionModel = IgnitionTimingMLP()
FuelRatioModel = FuelRatioMLP()


num_epochs = 100
datanum_per_epoch = 12000
batch_size = 128

criterion = nn.MSELoss()

IOptimizer = optim.AdamW(IgnitionModel.parameters(), lr=0.005, weight_decay=0.0001)
FROptimizer = optim.AdamW(FuelRatioModel.parameters(), lr=0.005, weight_decay=0.0001)

IScheduler = optim.lr_scheduler.CosineAnnealingLR(IOptimizer, T_max=num_epochs, eta_min=0.00001)
FRScheduler = optim.lr_scheduler.CosineAnnealingLR(FROptimizer, T_max=num_epochs, eta_min=0.00001)


def compute_rmse(y_true_tensor, y_pred_tensor,scaler):
    y_true_real = scaler.inverse_transform(y_true_tensor.cpu().numpy())
    y_pred_real = scaler.inverse_transform(y_pred_tensor.cpu().numpy())
    return np.sqrt(np.mean((y_true_real - y_pred_real)**2))

for epoch in range(num_epochs):
    train_dataset = SimEngineData(sample_count=datanum_per_epoch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    IgnitionModel.train()
    FuelRatioModel.train()

    for batch_X, batch_Ys, batch_Yf in train_loader:
        IgnitionModel.zero_grad()
        prediction_ignition = IgnitionModel(batch_X)
        loss_ignition = criterion(prediction_ignition, batch_Ys)
        loss_ignition.backward()
        IgnitionModel.step()

        FuelRatioModel.zero_grad()
        prediction_fuelratio = FuelRatioModel(batch_X)
        loss_fuelratio = criterion(prediction_fuelratio, batch_Yf)
        loss_ignition.backward()
        FuelRatioModel.step()

    IScheduler.step()
    FRScheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        IgnitionModel.eval()
        FuelRatioModel.eval()

        val_Ys_list, val_Yf_list = [], []
        pred_Ys_list, pred_Yf_list = [], []

        with torch.no_grad():
            for v_X, v_Ys, v_Yf in validation_loader:
                val_Ys_list.append(v_Ys)
                val_Yf_list.append(v_Yf)

                pred_Ys_list.append(IgnitionModel(v_X))
                pred_Yf_list.append(FuelRatioModel(v_X))

        all_val_Ys = torch.cat(val_Ys_list)
        all_val_Yf = torch.cat(val_Yf_list)
        all_pred_Ys = torch.cat(pred_Ys_list)
        all_pred_Yf = torch.cat(pred_Yf_list)


        val_mse_spark = criterion(all_pred_Ys, all_val_Ys).item()
        val_mse_fuel = criterion(all_pred_Yf, all_val_Yf).item()


        val_rmse_spark = compute_rmse(all_val_Ys, all_pred_Ys, scaler_Ys)
        val_rmse_fuel = compute_rmse(all_val_Yf, all_pred_Yf, scaler_Yf)

        current_lr =IScheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{num_epochs}] | LR: {current_lr:.5f}")
        print(f"  Spark Model -> Val MSE: {val_mse_spark:.5f} | Physical RMSE: {val_rmse_spark:.2f} °BTDC")
        print(f"  Fuel Model  -> Val MSE: {val_mse_fuel:.5f} | Physical RMSE: {val_rmse_fuel:.4f} λ\n")

IgnitionModel.eval()
FuelRatioModel.eval()










