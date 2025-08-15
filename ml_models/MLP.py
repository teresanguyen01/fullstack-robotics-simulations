import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import uuid

# Set random seed for reproducibility
torch.manual_seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Helper: drop Timestamp column
# -----------------------------
def drop_timestamp(df: pd.DataFrame, label: str) -> pd.DataFrame:
    drops = [c for c in df.columns if c.strip().lower() == "timestamp"]
    if drops:
        print(f"[{label}] Dropping columns: {drops}")
        df = df.drop(columns=drops)
    return df

# Load data

train_input_path = 'ml_modeling_0814/sensor_train.csv'
train_output_path = 'ml_modeling_0814/mocap_train.csv'
test_input_path = 'ml_modeling_0814/sensor_test.csv'
test_output_path = 'ml_modeling_0814/mocap_test.csv'

# Read as DataFrames so we can drop Timestamp safely
X_train_df = pd.read_csv(train_input_path)
y_train_df = pd.read_csv(train_output_path)
X_test_df  = pd.read_csv(test_input_path)
y_test_df  = pd.read_csv(test_output_path)

# Do NOT use Timestamp for training/testing
X_train_df = drop_timestamp(X_train_df, "X_train")
y_train_df = drop_timestamp(y_train_df, "y_train")
X_test_df  = drop_timestamp(X_test_df,  "X_test")
y_test_df  = drop_timestamp(y_test_df,  "y_test")

X_train = X_train_df.values
y_train_full = y_train_df.values
X_test = X_test_df.values
y_test_full = y_test_df.values

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_full.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test_full.shape}")

# Standardize input and output
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train_full)
y_test_scaled = scaler_y.transform(y_test_full)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

# Create DataLoaders
batch_size = 128  # Increased for efficiency; tune as needed
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define MLP model with ReLU
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
output_size = y_train_full.shape[1]
model = MLP(input_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)  # L2 reg
    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train R2: {train_r2:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}")

    # Scheduler step on test R2 (maximize)
    scheduler.step(test_r2)

    # Early stopping
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        no_improvement_count = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        no_improvement_count += 1

    if no_improvement_count >= patience:
        print("Early stopping triggered.")
        break

# Load best model
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Final evaluation
with torch.no_grad():
    y_train_pred_scaled = model(X_train_tensor)
    y_test_pred_scaled = model(X_test_tensor)

# Convert predictions back to original scale
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.cpu().numpy())
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.cpu().numpy())

# Calculate metrics
mse = mean_squared_error(y_test_scaled, y_test_pred_scaled.cpu().numpy())
train_r2 = r2_score(y_train_scaled, y_train_pred_scaled.cpu().numpy())

print("Mean Squared Error (MSE):", mse)
print("Training R2 Score:", train_r2)
print("Best Testing R2 Score:", best_test_r2)