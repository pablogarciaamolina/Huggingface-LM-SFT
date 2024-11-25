# HUGGINGFACE-LM-SFT

##### Supervised Fine Tunning over Language Models for *Instruction Following* with *Hugging Face*.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Usage](#usage)
   1. [Training](#training)
   2. [Evaluation](#evaluation) 
5. [Modules Description](#modules)


## Overview


This repository provides a simple framework for **Supervised Fine-Tuning (SFT)** of language models using Hugging Face libraries. It's purpose is to facilitate the interaction with pytho's Hugging Face API in order to fastly deploy, train, evaluate and try different strategies with multiple models.

Please note, this repository was developed as support for a final project. This means it has no active support and does not intend to cover all possible implementations.


## Project Structure

```plaintext
HUGGINGFACE-LM-SFT/
├── src/
├── instruction_following_eval/
│   │   ├── data/
│   │   │   ├── ...
│   │   ├── evaluation_main.py
│   │   ├── ...
│   ├── models/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── quantization.py
│   │   ├── sft.py
│   │   ├── tokenizers.py
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── resources_measuring.py
│   ├── train.py
│   ├── main.py
├── README.md
├── requirements.txt
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

As the size of this scope of this project is very small, only one file is proposed to run the desired code and all the abstraction is left behind in `\src`. In `main.py` one can find, right in the begining, some variables dictionaries that serve to quickly modify the hyparameters, set the model, choose the data, and other configurations.

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

To evaluate a trained model first make sure you have it saved locally and if the configuration in `main.py` is correct.

```bash
python main.py --evaluation
```

```bash
python -m main --evaluation
```

### Benchmarking

A way to measuring the performance of the model by the IFEVAL benchmark is provided. Make sure you have a trained model and check the configuration in `main.py`.

```bash
python main.py --measure
```

```bash
python -m main --measure
```

## Modules

In this section, a deeper explanation for each module of the repository is given. This is specially useful in the case of the **configurations**, as it will be explained how to modify the pipeline at will.

### `config.py`

This file stores the cofigurations variables of the proyect, which modify the pipeline for training, evaluating and measuring. These configurations variables will be listed and a brief explanation will be provided.

- `MODEL`: Specifies the type of model to use in the pipeline.

- `TOKENIZER`: The tokenizer instance or configuration. If `None`, the model's default tokenizer is used.

- `DATASET`: The dataset selected for training or evaluation.

- `EVALUATION_MODEL`: Path to the specific model checkpoint used for evaluation.

- `HYPERPARAMETERS`: Defines key parameters controlling the training process.
  - `learning_rate`: Initial learning rate for training.
  - `optim`: Optimizer used during training.
  - `per_device_train_batch_size`: Number of samples per device in each training batch.
  - `max_steps`: Total number of training steps.
  - `lr_scheduler_type`: Type of learning rate schedule.
  - `warmup_steps`: Steps during which the learning rate increases to its initial value.

- `TRAINER_SPECIFICATIONS`: Additional specifications for training behavior.
  - `max_seq_length`: Maximum sequence length allowed during training.

- `EVALUATION_SPECIFICATIONS`: Parameters controlling the evaluation process.
  - `batch_size`: Number of samples processed in each evaluation batch.
  - `num_batches`: Total number of evaluation batches.
  - `max_length`: Maximum token length for evaluation sequences.

- `TOKENIZER_CONFIG`: Options for customizing the tokenizer behavior.
  - `add_eos_token`: Whether to add an end-of-sequence token automatically.
  - `use_fast`: Whether to use the faster tokenizer implementation.
  - `padding_side`: Direction of padding for token sequences.

- `MODEL_CONFIG`: Settings for model initialization and behavior.
  - `device_map`: Strategy for distributing the model across devices.

- `DATASET_CONFIG`: Options for customizing dataset handling.
  - `num_val`: Number of validation samples. **Value**: `25`

- `USE_QUANTIZATION`: Flag indicating whether quantization is enabled.

- `QUANTIZATION_CONFIG`: Parameters for enabling model quantization for memory and computational efficiency.
  - `load_in_4bit`: Whether to load the model in 4-bit precision. **Value**: `True`
  - `bnb_4bit_quant_type`: Type of 4-bit quantization. **Value**: `"nf4"`
  - `bnb_4bit_compute_dtype`: Data type for computations during quantization. **Value**: `torch.bfloat16`
  - `bnb_4bit_use_double_quant`: Whether to use double quantization. **Value**: `True`

- `USE_LORA`: Indicates whether LoRA (Low-Rank Adaptation) fine-tuning is enabled.

- `LORA_CONFIG`: Configuration for LoRA fine-tuning.
  - `lora_alpha`: Scaling factor for LoRA updates. **Value**: `16`
  - `lora_dropout`: Dropout probability during LoRA training. **Value**: `0.05`
  - `r`: Rank of LoRA updates. **Value**: `16`
  - `bias`: Type of bias adjustment. **Value**: `"none"`
  - `task_type`: Task type for which LoRA is applied. **Value**: `TaskType.CAUSAL_LM`
  - `target_modules`: List of target modules to apply LoRA.

- `USE_PROMPT_TUNNING`: Indicates whether prompt tuning is enabled.

- `PROMPT_TUNNING_CONFIG`: Configuration for prompt tuning if enabled.
  - `task_type`: Task type for which prompt tuning is applied. **Value**: `TaskType.CAUSAL_LM`
  - `num_virtual_tokens`: Number of virtual tokens added during prompt tuning. **Value**: `50`


### `main.py`

This is the main file of the proyect. It runs the pipelines.
