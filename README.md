# HUGGINGFACE-LM-SFT

##### Supervised Fine Tunning over Language Models for *Instruction Following* with *Hugging Face*.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Usage](#usage)
   1. [Training](#training)
   2. [Evaluation](#evaluation) 
5. [Modules Description](#modules-description)


## Overview


This repository provides a simple framework for **Supervised Fine-Tuning (SFT)** of language models using Hugging Face libraries. It's purpose is to facilitate the interaction with pytho's Hugging Face API in order to fastly deploy, train, evaluate and try different strategies with multiple models.

Please note, this repository was developed as support for a final project. This means it has no active support and does not intend to cover all possible implementations.


## Project Structure

```plaintext
HUGGINGFACE-LM-SFT/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── quantization.py
│   │   ├── sft.py
│   │   ├── tokenizers.py
│   ├── __init__.py
│   ├── data.py
│   ├── resources_measuring.py
│   ├── train.py
│   ├── main.py
├── README.md
```

## Setup

### Prerequisites

Ensure you have the following:
- Python >= 3.8
- Set up connection with Hugging Face account.
- CUDA-compatible GPU and drivers (optional, for GPU acceleration).

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/pablogarciaamolina/Huggingface-LM-SFT
```
```bash
cd HUGGINGFACE-LM-SFT
```

### Install dependencies

```bash
pip install requirements.txt
```


## Usage

As the size of this scope of this project is very small, only one file is proposed to run the desired code and all the abstraction is left behind in `\src`. In `main.py` one can find, right in the begining, some dictionaries that serve to quickly modify the hyparameters, set the model, choose the data, and other configurations.

It is possible to make easy changes to the pipeline, just by modifying the functions with the different modules provided.

By default, the pipeline works with a model from the avaliable, applying quantization, and with LoRA and Prompt Tuning for **PEFT**.

Befor running the following commands, make sure the configuration is as expected (e.g. model, hyperparameters, ...)

### Training

```
python main.py --training
```

```
python -m main --training
```

### Evaluation

```bash
python main.py --evaluation
```

```bash
python -m main --evaluation
```