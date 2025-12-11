# wppkg

- A package I developed for my daily work.

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
pip install -e . [--no-deps]
```



