import argparse
import time
import torch
import flox

from pathlib import Path

from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from flox.data.utils import federated_split
from flox.flock.factory import create_standard_flock
from flox.nn import FloxModule

import parsl
from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor

from torchvision.models import alexnet, resnet18, resnet50, resnet152
from smallnet.smallnet import SmallNet

class Net(FloxModule):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.last_accuracy = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss


def main(args: argparse.Namespace):
    flock = create_standard_flock(num_workers=args.workers_nodes)
    root_dir = Path(args.root_dir)
    if "~" in str(root_dir):
        root_dir = root_dir.expanduser()
    data = FashionMNIST(
        root=str(root_dir),
        download=False,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        ),
    )
    fed_data = federated_split(
        data,
        flock,
        num_classes=10,
        labels_alpha=args.labels_alpha,
        samples_alpha=args.samples_alpha,
    )

    parsl_local = {"label": "local-htex", "max_workers_per_node": args.workers_nodes}
    parsl_remote  = {
            "label" : "expanse-htex",
            "max_workers_per_node": args.workers_nodes, # NOTE (nathaniel-hudson): replace with your config
            "provider": SlurmProvider(
                'debug',
                account='TG-CCR180005',
                launcher=SrunLauncher(),
                scheduler_options='', ##SBATCH ntasks-per-node=128',
                worker_init='/bin/bash; ~/miniconda3/bib/conda activate flox-test; export PYTHONPATH="/home/chard/flox-scaling-tests/parsl-tests"',
                walltime='00:30:00',
                init_blocks=1,
                max_blocks=1,
                min_blocks=1,
                nodes_per_block=1)}

    flox.federated_fit(
        flock=flock,
        module=Net(),  # nathaniel-hudson: this uses a reasonable model.
        #module=None, # nathaniel-hudson: this uses a VERY small debug model.
        #module=KyleNet(),
        #module=resnet18(weights=None),
        datasets=fed_data,
        num_global_rounds=args.rounds,
        strategy="fedsgd",
        kind="sync",
        debug_mode=True,
        launcher_kind=args.executor,
        launcher_cfg=parsl_local
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--executor",
        "-e",
        type=str,
        choices=["process", "thread", "parsl", "globus-compute"],
        default="parsl",
    )
    args.add_argument("--max_workers", "-w", type=int, default=1)
    args.add_argument("--workers_nodes", "-n", type=int, default=2)
    args.add_argument("--samples_alpha", "-s", type=float, default=1000.0)
    args.add_argument("--labels_alpha", "-l", type=float, default=1000.0)
    args.add_argument("--rounds", "-r", type=int, default=1)
    args.add_argument("--root_dir", "-d", type=str, default=".")
    parsed_args = args.parse_args()
    assert parsed_args.samples_alpha > 0.0
    assert parsed_args.labels_alpha > 0.0
    start_time = time.perf_counter()
    print(f"start:{start_time}")
    main(parsed_args)
    end_time = time.perf_counter()
    print(f"end:{end_time}")
    print(f"❯ Finished in {end_time - start_time} seconds.")
