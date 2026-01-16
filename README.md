<p align="center">
  <img src="assets/logo.png" alt="wppkg logo" width="240">
</p>

<p align="center">
  <a href="https://pypi.org/project/wppkg/"><img  alt="PyPI version" src="https://img.shields.io/pypi/v/wppkg.svg?color=purple"></a>
  <a href="https://pepy.tech/project/wppkg"><img alt="PyPI total downloads" src="https://pepy.tech/badge/wppkg"></a>
  <a href="https://github.com/Peg-Wu/wppkg/blob/main/LICENSE"><img alt="Github" src="https://img.shields.io/github/license/Peg-Wu/wppkg.svg?color=green"></a>
  <a href="https://github.com/Peg-Wu/wppkg/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/Peg-Wu/wppkg.svg?color=orange"></a>
</p>

wppkg is a package I developed for my daily work.

<details>
<summary>About Me</summary>

My main research interests are <b>Bioinformatics</b> and <b>Artificial Intelligence</b>.
Specifically, my main research direction is <b>AI Virtual Cell (AIVC)</b>, including:

- <b>Building foundation models for multi-omics</b>
- <b>Single-cell perturbation prediction</b>

Besides, I am also interested in Reinforcement Learning (RL) and AI Agent Application Development. I’m passionate about bringing cutting-edge AI techniques into single-cell biology to accelerate the development of AIVC.

> [!TIP]
> VCs must work across biological scales, over time, and across data modalities and should help reveal the programming language of cellular systems and provide an interface to use it for engineering purposes.

</details>

---

## Installation

### Environment Setup

### Step 1: Set up a python environment

We recommend creating a virtual Python environment with [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/):

- Required version: `python >= 3.10`

```bash
conda create -n wppkg python=3.10
conda activate wppkg
```

### Step 2: Install pytorch

Install `PyTorch` based on your system configuration. Refer to [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) 

For the exact command, for example:

- You may choose any version to install, but make sure the PyTorch version is not too old.
- We recommend `torch ≥ 2.6`.

```bash
# Installation Example: torch v2.8.0
# CUDA 12.6
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# CUDA 12.9
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

### Step 3: Install deepspeed (optional)

Install `DeepSpeed` based on your system configuration. Refer to [DeepSpeed installation instructions](https://www.deepspeed.ai/tutorials/advanced-install/) 

For the exact command, for example:

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
pip install [-e] .

# w/o dependencies
pip install [-e] . --no-deps
```

### Update wppkg

If you want to update all dependencies of `wppkg` except `torch`, you can run the following command:

```bash
pip install -U $(pip show wppkg | sed -n 's/^Requires: //p' | tr ',' ' ' | xargs -n1 | grep -vi '^torch$')
```

---

## Trainer Tips

- Early stopping does not currently support resuming training. If training is forcibly resumed, the early stopping callback will be reinitialized.
- If you enable early stopping, ensure that `eval_every_n_epochs` and `checkpointing_steps` are aligned, as the Trainer does not automatically save the best model. 
- The final model is always saved at the end of training, even if early stopping is triggered.

---

## Star History

<p align="center">
  <a href="https://star-history.com/#Peg-Wu/wppkg&Date">
    <img 
      src="https://api.star-history.com/svg?repos=Peg-Wu/wppkg&type=Date"
      width="600"
      alt="Star History Chart"
    />
  </a>
</p>
