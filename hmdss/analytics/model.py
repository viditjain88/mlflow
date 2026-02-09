import torch
import torch.nn as nn

class PatientVolumePredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16):
        super(PatientVolumePredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1) # Predicts 1 value: patient volume

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x
