# Flight Artifacts

This repository presents the source code for the experimental tests for the following paper:
> Hudson, N., Hayot-Sasson, V., Babuji, Y., Baughman, M., Pauloski, J. G., Chard, R., ... & Chard, K. (2024). Flight: A FaaS-Based Framework for Complex and Hierarchical Federated Learning. arXiv preprint arXiv:2409.16495.

```bibtex
@article{hudson2024flight,
  title={Flight: A FaaS-Based Framework for Complex and Hierarchical Federated Learning},
  author={Hudson, Nathaniel and Hayot-Sasson, Valerie and Babuji, Yadu and Baughman, Matt and Pauloski, J Gregory and Chard, Ryan and Foster, Ian and Chard, Kyle},
  journal={arXiv preprint arXiv:2409.16495},
  year={2024}
}
```

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
Weak-scaling tests using our proposed Flight framework on HPC systems with Parsl.
These tests use Parsl's default data transfer implementation in addition to Redis (via Proxystore) as separate tests.

#### Artifact 1.2: Flower Scaling Tests
Weak-scaling tests for the Flower framework.

### Artifact 2: Hierarchy Simulation Test
Tests that simulate hierarchical federated learning with Flight.
Calculations of communication costs are also included.

### Artifact 3: Asynchronous Simulation Test
Tests that compare synchronous and asynchronous federated learning with Flight.

### Artifact 4: Remote EC2 Test
Remote execution tests prepared for Amazon EC2 instances.

