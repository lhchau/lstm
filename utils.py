"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
import yaml
import torch
from ast import literal_eval
from torch.utils.data import DataLoader, Dataset

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

# Create sequences (input windows)
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences