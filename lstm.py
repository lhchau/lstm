import os
import wandb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import *
import matplotlib.pyplot as plt

##########################################
# Hypeparameter Preparation
##########################################

# Data hyperparameter
cfg = {
    'data_name': 'yogurt',
    
    'val_size': 100,
    'sequence_length': 6, # Number of days for each input sequence
    'batch_size': 16,

    # LSTM hyperparameter
    'hidden_size': 64,
    'num_layers': 2,

    # Optimization hyper
    'num_epochs': 200,
    'learning_rate': 0.001,
    
    # Logging framework, wandb
    'project_name': None,
}
cfg = exec_configurator(cfg)
num_epochs = cfg['num_epochs']
data_name = cfg['data_name']
best_val_loss = float('inf')  # Set to infinity initially
best_model_path = os.path.join('checkpoint', 'best_model.pth')
if not os.path.isdir('checkpoint'):
    os.makedirs('checkpoint', exist_ok=True)
    
if cfg['project_name'] is not None:
    logging_name = ''
    for k, v in cfg.items():
        if '_' in k:
            pre_fix, post_fix = k.split('_')
            logging_name += (pre_fix[:2] + '_' + post_fix[:2] + '=' + str(v) + ';')
    wandb.init(project=cfg['project_name'], name=logging_name, config=cfg)
##########################################
# Data Preparation
##########################################
df = pd.read_csv(os.path.join('.', 'data', f'{data_name}.csv'))
df['Quantity'] = df['Quantity'].str.replace(',', '').str.strip().astype(float)

train_df = df[:-cfg['val_size'] + cfg['sequence_length']]
val_df = df[-cfg['val_size']:]

mean = train_df['Quantity'].mean()
std = train_df['Quantity'].std()
train_df['Quantity'] = (train_df['Quantity'] - mean) / std
val_df['Quantity'] = (val_df['Quantity'] - mean) / std

train_data = create_sequences(train_df['Quantity'].values, cfg['sequence_length'])
val_data = create_sequences(val_df['Quantity'].values, cfg['sequence_length'])

# Create a DataLoader
train_dataset = TimeSeriesDataset(train_data)
val_dataset = TimeSeriesDataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

##########################################
# Define the LSTM model
##########################################
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get the output of the last time step
        out = self.fc(lstm_out)
        return out

# Instantiate the model, define the loss function and optimizer
model = LSTMModel(hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

for epoch in range(cfg['num_epochs']):
    total_loss = 0
    
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.unsqueeze(-1)  # Add input size dimension
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss/(batch_idx + 1):.4f}')

    val_total_loss = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(val_loader):
            sequences = sequences.unsqueeze(-1)  # Add input size dimension
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_total_loss += loss.item()
            
    val_avg_loss = val_total_loss / (batch_idx + 1)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validaation Loss: {val_avg_loss:.4f}')
    # Save best model based on validation loss
    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
    
    wandb.log({
        'train/loss': total_loss,
        'val/loss': val_total_loss,
        'val/best_loss': best_val_loss
    })
    
##########################################
# Function to visualize predictions
##########################################
def visualize_predictions(model, val_loader, sequence_length, mean, std):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.unsqueeze(-1)  # Add input size dimension
            outputs = model(sequences)
            predictions.extend(outputs.squeeze().numpy())
            actuals.extend(labels.numpy())

    # Denormalize the predictions and actuals
    predictions = np.array(predictions) * std + mean
    actuals = np.array(actuals) * std + mean

    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Actual vs Predicted Values')
    plt.xlabel(f'Time Steps (each step = {sequence_length} days)')
    plt.ylabel('Quantity')
    plt.legend()
    # Save the figure to log on wandb
    plt_path = "predictions_vs_actuals.png"
    plt.savefig(plt_path)

    # Log the plot on wandb
    wandb.log({"Predictions vs Actuals": wandb.Image(plt_path)})

# Load best model for visualization
model.load_state_dict(torch.load(best_model_path))
visualize_predictions(model, val_loader, cfg['sequence_length'], mean, std)