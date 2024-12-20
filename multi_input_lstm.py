import os
import yaml
import sys
import wandb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ast import literal_eval
from torch.utils.data import DataLoader, Dataset

##########################################
# Utilization
##########################################

# Define a PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return torch.Tensor(sequence), torch.Tensor([label])

def exec_configurator(cfg):
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            yaml_config_file = arg
            print(f"Overriding config with {yaml_config_file}:")
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            idx = arg.find('=')
            key, val = arg[2:idx], arg[idx+1:]
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            # assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            cfg[key] = attempt
    return cfg

# Create sequences (input windows (sequence length))
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length][0]
        sequences.append((seq, label))
    return sequences

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
    'hidden_size': 128,
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
df = df.drop(columns=['Date'])
##########################################
### Data preprocessing
##########################################
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].str.replace(',', '').str.strip().astype(float)
    
train_df = df[:-cfg['val_size'] + cfg['sequence_length']]
val_df = df[-cfg['val_size']:]

not_normalized_columns = ['weather'] 

mean_values = df.drop(columns=not_normalized_columns).mean()
std_values = df.drop(columns=not_normalized_columns).std()

for column, mean, std in zip(mean_values.index, mean_values.values, std_values.values):
    train_df[column] = (train_df[column] - mean) / std
    val_df[column] = (val_df[column] - mean) / std

train_df, val_df = train_df.to_numpy(), val_df.to_numpy()

train_data = create_sequences(train_df, cfg['sequence_length'])
val_data = create_sequences(val_df, cfg['sequence_length'])

##########################################
### Create a DataLoader
##########################################
train_dataset = TimeSeriesDataset(train_data)
val_dataset = TimeSeriesDataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

##########################################
# Define the LSTM model
##########################################
class LSTMModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=50, num_layers=2, output_size=1):
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
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_total_loss += loss.item()
            
    val_avg_loss = val_total_loss / (batch_idx + 1)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_avg_loss:.4f}')
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
visualize_predictions(model, val_loader, cfg['sequence_length'], mean_values.iloc[0], std_values.iloc[0])