from typing import List, Tuple
from time import time
import flwr as fl
from flwr.common import Metrics
import numpy as np
import argparse
from torchvision.models import resnet18

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from torchvision.models import resnet18, resnet50, resnet152


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


model = Net()


def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


# Get initial model parameters
initial_model_parameters = get_parameters(model)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--clients", required=True, type=int, help="Number of clientsi")

parser.add_argument("--ip", required=True, type=str, help="ip address")
num_clients = parser.parse_args().clients
ip = parser.parse_args().ip


# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_available_clients=num_clients,
    min_fit_clients=num_clients,
    initial_parameters=initial_model_parameters,
)


# Start Flower server
fl.server.start_server(
    server_address=ip + ":9898",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)
print("hi")
end = time()
print(end)
with open("flower_test_" + str(num_clients) + "_" + str(18) + ".txt", "w") as f:
    f.write(str(end))
