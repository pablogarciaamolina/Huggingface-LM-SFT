import os

import torch

from .data import FT_training_dataset
from src.models import (
    ModelType,
    LoRALoader,
    PromptTunningLoader,
    PEFTLoader,
    QuantizationLoader,
    ModelLoader,
    TokenizerLoader
)
from src.data import ALPACA, LIMA, ULTRABIN
from src.train import Trainer

RESULTS_PATH = "models"

MODEL = ModelType.PHI_2
TOKENIZER = None
DATASET = ALPACA

HYPERPARAMETERS = {
    "learning_rate": 1e-4,                   # Initial learning rate
    "optim": "paged_adamw_8bit",             # Use 8-bit AdamW optimizer for memory efficiency
    "per_device_train_batch_size": 4,        # Batch size per device during training
    "max_steps": 200,                        # Total number of training steps
    "lr_scheduler_type": "linear",            # Use a linear learning rate scheduler
    "warmup_steps": 25,                      # Number of warmup steps for learning rate scheduler
}

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

QUANTIZATION_CONGIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": getattr(torch, "bfloat16"),
    "bnb_4bit_use_double_quant": True
}

def train():
    """
    Pipeline for SFT on a pretrained model
    """

    # -------------------- MODIFY TO CHANGE PIPELINE --------------------
    # Prepare quantization scheme
    quantization = QuantizationLoader(**QUANTIZATION_CONGIG).load()
    # quantization = None

    # Set PEFT
    lora = LoRALoader().load()
    prompt_tunning = PromptTunningLoader(MODEL, num_virtual_tokens=50).load()
    peft = PEFTLoader([lora, prompt_tunning]).load()

    # -------------------------------------------------------------------
    # ---------------------- NO NEED TO MODIFY --------------------------
    # Load Tokenizer
    tokenizer = TokenizerLoader(TOKENIZER if TOKENIZER else MODEL, **TOKENIZER_CONFIG).load()
    # Load Model
    model = ModelLoader(MODEL, pad_token_id=tokenizer.pad_token_id, quantization=quantization, **MODEL_CONFIG).load()
    # Load dataset
    dataset = FT_training_dataset(dataset_name=DATASET, **DATASET_CONFIG)

    # Prepare trainer
    trainer = Trainer(
        model,
        dataset,
        tokenizer,
        HYPERPARAMETERS,
        peft,
        os.path.join(RESULTS_PATH, MODEL+"_results")
    )

    # Train
    trainer.train()


if __name__ == "__main__":

    train()