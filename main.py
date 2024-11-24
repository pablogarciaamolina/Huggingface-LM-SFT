import os
import argparse
import time

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
from src.train import Trainer, RESULTS_PATH
from src.evaluate import IFEVALEvaluator, SAVING_EVALUATION_DIR, SAVING_MEASURE_DIR

MODEL = ModelType.PHI_2
TOKENIZER = None
DATASET = ALPACA

EVALUATION_MODEL = "microsoft-phi-2_ultrabin_lora_results/checkpoint-75"

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
    "max_seq_length": 512
}

EVALUATION_SPECIFICATIONS = {
    "batch_size": 8,
    "num_batches": 2,
    "max_length": 128
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
    peft_config = lora

    # -------------------------------------------------------------------
    # ---------------------- NO NEED TO MODIFY --------------------------
    # Load Tokenizer
    tokenizer = TokenizerLoader(TOKENIZER if TOKENIZER else MODEL, **TOKENIZER_CONFIG).load()
    # Load Model - Add PEFT
    model = ModelLoader(MODEL, pad_token_id=tokenizer.pad_token_id, quantization=quantization, **MODEL_CONFIG).load()
    # Load dataset
    dataset = FT_training_dataset(dataset_name=DATASET, tokenizer=tokenizer, **DATASET_CONFIG)

    # Prepare trainer
    trainer = Trainer(
        model,
        dataset,
        tokenizer,
        peft_config,
        HYPERPARAMETERS,
        MODEL.replace("/", "-")+f"{time.time()}_results",
        True,
        **TRAINER_SPECIFICATIONS
    )

    # Train
    trainer.train()

def evaluation() -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = os.path.join(RESULTS_PATH, EVALUATION_MODEL)
    tokenizer = TokenizerLoader(path).load()
    model = ModelLoader(path, pad_token_id=tokenizer.pad_token_id, device_map=device).load()

    evaluator = IFEVALEvaluator(model, tokenizer, verbose=True)
    evaluator.evaluate(
        EVALUATION_MODEL.replace("/", "-"),
        device,
        **EVALUATION_SPECIFICATIONS
    )

def ifeval() -> None:

    results = IFEVALEvaluator.measure(EVALUATION_MODEL.replace("/", "-"))

    print(results.stdout)
    print(results.stderr)


if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run training or evaluation")
    parser.add_argument('--train', action='store_true', help="Run training")
    parser.add_argument('--evaluation', action='store_true', help="Run evaluation")
    parser.add_argument('--measure', action='store_true', help="Run IFEVAL benchmark")

    # Parse the arguments
    args = parser.parse_args()

    if args.train:
        print("Running training...")
        train()
    elif args.evaluation:
        print("Running evaluation...")
        evaluation()
    elif args.measure:
        print("Running benchmarking...")
        ifeval()
    else:
        parser.print_help()