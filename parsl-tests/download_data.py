import argparse

from torchvision.datasets import FashionMNIST

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--root_dir", "-d", type=str, default=".")
    args = args.parse_args()

    for train in [True, False]:
        # Download both the training and testing data for good measure.
        FashionMNIST(
            root=args.root_dir,
            download=True,
            train=train,
        )
