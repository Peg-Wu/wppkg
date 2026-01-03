<p align="center">
  <img src="assets/logo.png" alt="wppkg logo" width="120">
</p>

<p align="center">
  <a href="https://pypi.org/project/wppkg/">
    <img src="https://img.shields.io/pypi/v/wppkg.svg" alt="PyPI version">
  </a>
  <a href="https://pepy.tech/project/wppkg">
    <img src="https://pepy.tech/badge/wppkg" alt="Total downloads">
  </a>
</p>

wppkg is a package I developed for my daily work.

---

## Installation

### Environment Setup

### Step 1: Set up a Python environment

We recommend creating a virtual Python environment with [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/):

```bash
conda create -n wppkg python=3.10
conda activate wppkg
```

### Step 2: Install Pytorch

Install `PyTorch` based on your system configuration. Refer to [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) for the exact command. For example:

- You may choose any version to install, but make sure the PyTorch version is not too old.

```bash
# Installation Example: torch v2.8.0
# CUDA 12.6
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# CUDA 12.9
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

### Step 3: Install DeepSpeed (Optional)

Install `DeepSpeed` based on your system configuration. Refer to [DeepSpeed installation instructions](https://www.deepspeed.ai/tutorials/advanced-install/) for the exact command. For example:

```bash
pip install deepspeed
```

### Step 4: Install wppkg and dependencies

To install `wppkg`, run:

```bash
pip install wppkg
```

Or install from github:

```python
git clone https://github.com/Peg-Wu/wppkg
cd wppkg
pip install -e .

# w/o dependencies
pip install -e . --no-deps
```

### Update wppkg

If you want to update all dependencies of `wppkg` except `torch`, you can run the following command:

```bash
pip install -U $(pip show wppkg | sed -n 's/^Requires: //p' | tr ',' ' ' | xargs -n1 | grep -vi '^torch$')
```

## Trainer Tips

- Early stopping does not currently support resuming training. If training is forcibly resumed, the early stopping callback will be reinitialized.
- If you enable early stopping, ensure that `eval_every_n_epochs` and `checkpointing_steps` are aligned, as the Trainer does not automatically save the best model. 
- The final model is always saved at the end of training, even if early stopping is triggered.

