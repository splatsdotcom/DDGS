# DDGS
This is the home of the differentiable renderer for the **Dynamoc Gaussian Splat** (`.dgs`) file format. This repository contains:
- A CUDA-accerated, differentiable renderer for dynamic gaussian splats.
- Utilties for data loading and initialization of dynamic gaussian splats.

This library serves as core infrastructure powering all of the models within [Splatkit](https://github.com/splatsdotcom/splatkit). All of Splatkit's gaussian training utilities use DDGS as their renderer.

## Building + Usage
To use `DDGS` within your own project, you will have to build it from source. A `pip` install is coming soon! First, ensure you have the necessary dependencies:
- A [CUDA toolkit](https://developer.nvidia.com/cuda/toolkit) compatible with your system's CUDA drivers
- A version of `pytorch`, compiled with the same CUDA version as your CUDA toolkit. See [here](https://pytorch.org/get-started/locally/).

Then, you can clone the repository and initialize the submodules:
```bash
git clone git@github.com:splatsdotcom/DDGS.git
cd DDGS
git submodule update --init --recursive
```
The project can then be built with:
```bash
pip install --no-build-isolation -e .
```
Now, `ddgs` will be available system-wide, it can be imported with a simple:
```python
import ddgs
```
See [example.py](https://github.com/splatsdotcom/DDGS/blob/main/example.py) for an example of how to setup `ddgs` as well as train your first splats.

## Documentation
Coming soon!