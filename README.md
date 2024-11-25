# HUGGINGFACE-LM-SFT

##### Supervised Fine Tunning over Language Models for *Instruction Following* with *Hugging Face*.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Usage](#usage)
   1. [Training](#training)
   2. [Evaluation](#evaluation)
   3. [Benchmarking](#benchmarking)
5. [Supported Models and Datasets](#supported-models-and-datsets)
   1. [Models](#models)
   2. [Datasets](#datasets)
6. [Modules Description](#modules)
   1. [config.py](#configpy)
   2. [main.py](#mainpy)
   3. [src](#src)


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

## Supported Models and Datsets

### Models

The following models are supported for every one of the usage cases.

- `Qwen/Qwen2.5-7B`
- `mistralai/Mistral-7B-v0.3`
- `Qwen/Qwen2.5-7B`
- `meta-llama/Llama-3.1-8B`
- `google/gemma-2-2b-jpn-it`
- `google/recurrentgemma-9b`

### Datasets

The following datasets are supported and have their individual mappings implemented.

- `tatsu-lab/alpaca`
- `ContextualAI/ultrabin_clean_max_chosen_min_rejected_rationalized_instruction_following`
- `GAIR/lima`



## Modules

In this section, a deeper explanation for each module of the repository is given. This is specially useful in the case of the **configurations**, as it will be explained how to modify the pipeline at will.

### `config.py`

This file stores the cofigurations variables of the proyect, which modify the pipeline for training, evaluating and measuring. These configurations variables will be listed and a brief explanation will be provided.

- `MODEL`: Specifies the model to use.

- `TOKENIZER`: The tokenizer to use. If `None`, the model's tokenizer is used.

- `DATASET`: The dataset selected for training.

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
  - `num_val`: Number of validation samples.

- `USE_QUANTIZATION`: Flag indicating whether quantization is enabled.

- `QUANTIZATION_CONFIG`: Parameters for enabling model quantization for memory and computational efficiency.
  - `load_in_4bit`: Whether to load the model in 4-bit precision.
  - `bnb_4bit_quant_type`: Type of 4-bit quantization.
  - `bnb_4bit_compute_dtype`: Data type for computations during quantization.
  - `bnb_4bit_use_double_quant`: Whether to use double quantization.

- `USE_LORA`: Indicates whether LoRA fine-tuning is enabled.

- `LORA_CONFIG`: Configuration for LoRA fine-tuning.
  - `lora_alpha`: Scaling factor for LoRA updates.
  - `lora_dropout`: Dropout probability during LoRA training. **Valu
  - `r`: Rank of LoRA updates.
  - `bias`: Type of bias adjustment.
  - `task_type`: Task type for which LoRA is applied. **Valu
  - `target_modules`: List of target modules to apply LoRA.

- `USE_PROMPT_TUNNING`: Indicates whether prompt tuning is enabled.

- `PROMPT_TUNNING_CONFIG`: Configuration for prompt tuning if enabled.
  - `task_type`: Task type for which prompt tuning is applied.
  - `num_virtual_tokens`: Number of virtual tokens added during prompt tuning.

Note that all parameters follow Hugging Face specifications. For further information, visit the oficial web site.

### `main.py`

This is the main file of the proyect. It runs the pipelines. No modifications are needed in this file, it should only be used when calling the different steps.

### `src`

#### instruction_following_eval

This is the module that computes IFEVAL benchmarking. It contains all needed functionalities as well as some required JSON files.

#### models

This module contains the abstractions to Hugging Face API. It contains a bunch of objects that facilitate our pipeline.

#### data.py

This module contains the abstractions of the supported Datasets. This include their mapping functions wich process the raw data with the tokenizer.

#### evaluate.py

This module wraps the methods needed for running evaluation under `"google/IFEval"` dataset and storing the result in a JSON file.

It also has a mathod to run the IFEVAL benchmarking over this results.

#### resources_measuring.py

This module contains functions to measure the computational requirements a model is using.

It is supposed to contain methods to estimate the load a process would require before running it (useful to know whether the GPU memory will be suficient), but this has not been implemented yet.

These methods have not been implemented for quick deployment yet.

#### train.py

This contains the abstraction of the SFTTrainer() from Hugging Face. In addition to the configurations from the configuration file, at the beggining of the script there is a dictionary for further configure the training. This dictionary contains parameters to a deper level and do not need to be changed unless a specific training is desired.
