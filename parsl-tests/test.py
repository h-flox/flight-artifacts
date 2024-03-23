import argparse
import flox

from flox.flock import Flock
from flox.flock.factory import create_standard_flock
from flox.data.utils import federated_split
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from pathlib import Path

import parsl
from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor

def main(args: argparse.Namespace):
    flock = create_standard_flock(num_workers=args.workers_nodes)
    root_dir = Path(args.root_dir)
    if "~" in str(root_dir):
        root_dir = root_dir.expanduser()
    data = FashionMNIST(
        root=root_dir,
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
    flox.federated_fit(
        flock=flock,
        module=None,  # NOTE (nathaniel-hudson): Okay because debug mode provides a fixed model.
        datasets=fed_data,
        num_global_rounds=args.rounds,
        strategy="fedsgd",
        kind="sync",
        debug_mode=True,
        launcher_kind=args.executor,
        launcher_cfg={
            "label": "parsl", 
             "max_workers_per_node": args.workers_nodes, # NOTE (nathaniel-hudson): replace with your config
             "provider": SlurmProvider(
               'debug',
               account='TG-CCR180005',
               launcher=SrunLauncher(),
               scheduler_options='', ##SBATCH ntasks-per-node=128',
               worker_init='/bin/bash; ~/miniconda3/bib/conda activate flox-test',
               walltime='00:30:00',
               init_blocks=1,
               max_blocks=1,
               min_blocks=1,
               nodes_per_block=1)
        }
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
    main(parsed_args)
