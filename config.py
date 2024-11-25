import torch
from peft import TaskType

from src.models import ModelType
from src.data import (
    ALPACA,
    LIMA,
    ULTRABIN
)
from src.models import BASE_MODULES

# PATHS
MODEL = ModelType.PHI_2
TOKENIZER = None
DATASET = ALPACA

EVALUATION_MODEL = "microsoft-phi-2_ultrabin_results/checkpoint-75"

# HYPERPARAMETERS
HYPERPARAMETERS = {
    "learning_rate": 1e-4,                   # Initial learning rate
    "optim": "paged_adamw_8bit",             # Use 8-bit AdamW optimizer for memory efficiency
    "per_device_train_batch_size": 4,        # Batch size per device during training
    "max_steps": 200,                        # Total number of training steps
    "lr_scheduler_type": "linear",            # Use a linear learning rate scheduler
    "warmup_steps": 25,                      # Number of warmup steps for learning rate scheduler
}

# TOKENIZER, MODEL and DATASET
TOKENIZER_CONFIG = {
    "add_eos_token": True,
    "use_fast": True,
    "padding_side": "left"
}
MODEL_CONFIG = {
    "device_map": "auto"
}
DATASET_CONFIG = {
    "num_val": 25
}

# QUANTIZATION
USE_QUANTIZATION: bool = True
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": getattr(torch, "bfloat16"),
    "bnb_4bit_use_double_quant": True
}

# PEFT
USE_LORA: bool = True
LORA_CONFIG = {
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "r": 16,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
    "target_modules": BASE_MODULES
}

USE_PROMPT_TUNNING: bool = False
PROMPT_TUNNING_CONFIG = {
    "task_type": TaskType.CAUSAL_LM,
    "num_virtual_tokens": 50
}

# TRAINING
TRAINER_SPECIFICATIONS = {
    "max_seq_length": 512
}

# EVALUATION
EVALUATION_SPECIFICATIONS = {
    "batch_size": 8,
    "num_batches": 2,
    "max_length": 128
}