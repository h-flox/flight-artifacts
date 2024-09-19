# Flight Artifacts

This repository presents the source code for the **scaling tests for the following paper**:
> John Smith, et al. "{title goes here}." Some Venue (YYYY).

These scaling tests involve benchmarks using our own **Flight** federated learning framework and the
[**Flower**](https://flower.ai) framework.

## General Setup

### Python
First, you must setup your Python envrionment (using either `venv` or `conda`). 
These tests were run with Python 3.11.8 specifically.
To set 

```sh
$ conda create -n=<env_name>  python=3.11.8
$ conda activate <env_name>
```

or

```sh
$ python3.11 -m venv <env_name>
$ source <env_name>/bin/activate
```

### Downloading FashionMNIST Data
For any of the model training in these tests, we use the Fashion MNIST benchmark dataset. To use this dataset for these tests,
you must first download the data onto your machine in a directory of your choosing (just be sure to take note of where you save it).
This can be done by running the provided Python script:
```sh
$ python download_data.py --root .
```
This will download the dataset using `torchvision.datasets`. 

## Artifacts

### Artifact 1: Scaling Tests

#### Artifact 1.1: Flight Scaling Tests with Parsl
...

#### Artifact 1.2: Flower Scaling Tests
...

### Artifact 2: Hierarchy Simulation Test
...

### Artifact 3: Asynchronous Simulation Test
...

### Artifact 4: Remote EC2 Test
...

