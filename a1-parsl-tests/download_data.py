import argparse

from torchvision.datasets import FashionMNIST

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--root_dir", "-d", type=str, default=".")
    args = args.parse_args()

    FashionMNIST(
        root=args.root_dir,
        download=True,
        train=True,
    )
