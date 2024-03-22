# Getting Started
To install FLoX, run the following in your environment:
```sh
pip install git+git+https://github.com/nathaniel-hudson/flox
```

## Pydantic versioning issue
There is a version for on `pydantic` between two dependencies:
1. `globus_compute_sdk`
2. `proxystore`

The easiest solution to get around this is to _first_ install the necessary libraries via
```sh
pip install -r requirements.txt
```

Then, you will want to install `proxystore` via:
```sh
pip install "proxystore[all]"
```

Then end with:
```sh
pip install pydantic==2.6.3
```
> This should already have been done from installing `proxystore`, but just for good measure.

***

As far as we can see, this does not cause any actual issues. There will be an annoying `UserWarning` 
thrown by `pydantic` from `funcx_common`, but no code-breaking issues.