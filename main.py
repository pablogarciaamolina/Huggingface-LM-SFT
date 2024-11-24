import os
import argparse

import torch

from src.data import FT_training_dataset
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

QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": getattr(torch, "bfloat16"),
    "bnb_4bit_use_double_quant": True
}

TRAINER_SPECIFICATIONS = {
    "max_seq_len": 512
}

def train():
    """
    Pipeline for SFT on a pretrained model
    """

    # -------------------- MODIFY TO CHANGE PIPELINE --------------------
    # Prepare quantization scheme
    quantization = QuantizationLoader(**QUANTIZATION_CONFIG).load()
    # quantization = None

    # Set PEFT
    lora = LoRALoader().load()
    prompt_tunning = PromptTunningLoader(MODEL, num_virtual_tokens=50).load()
    peft_config = [lora, prompt_tunning]

    # -------------------------------------------------------------------
    # ---------------------- NO NEED TO MODIFY --------------------------
    # Load Tokenizer
    tokenizer = TokenizerLoader(TOKENIZER if TOKENIZER else MODEL, **TOKENIZER_CONFIG).load()
    # Load Model - Add PEFT
    model = ModelLoader(MODEL, pad_token_id=tokenizer.pad_token_id, quantization=quantization, **MODEL_CONFIG).load()
    model = PEFTLoader(peft_config).load(model)
    # Load dataset
    dataset = FT_training_dataset(dataset_name=DATASET, tokenizer=tokenizer, **DATASET_CONFIG)

    # Prepare trainer
    trainer = Trainer(
        model,
        dataset,
        tokenizer,
        HYPERPARAMETERS,
        os.path.join(RESULTS_PATH, MODEL+"_results"),
        True,
        **TRAINER_SPECIFICATIONS
    )

    # Train
    trainer.train()

def evaluation() -> None:

    print("nothing")


if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run training or evaluation")
    parser.add_argument('--train', action='store_true', help="Run training")
    parser.add_argument('--evaluation', action='store_true', help="Run evaluation")

    # Parse the arguments
    args = parser.parse_args()

    if args.train:
        print("Running training...")
        train()
    elif args.evaluation:
        print("Running evaluation...")
        evaluation()
    else:
        parser.print_help()