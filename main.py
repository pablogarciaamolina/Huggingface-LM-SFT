import os
import argparse
import time

import torch

from src.data import FT_training_dataset
from src.models import (
    LoRALoader,
    PromptTunningLoader,
    PEFTLoader,
    QuantizationLoader,
    ModelLoader,
    TokenizerLoader
)
from src.train import Trainer, RESULTS_PATH
from src.evaluate import IFEVALEvaluator

from config import (
    QUANTIZATION_CONFIG,
    USE_QUANTIZATION,
    USE_LORA,
    LORA_CONFIG,
    USE_PROMPT_TUNNING,
    PROMPT_TUNNING_CONFIG,
    TOKENIZER,
    TOKENIZER_CONFIG,
    MODEL,
    MODEL_CONFIG,
    DATASET,
    DATASET_CONFIG,
    HYPERPARAMETERS,
    TRAINER_SPECIFICATIONS,
    EVALUATION_MODEL,
    EVALUATION_SPECIFICATIONS,

)

def train():
    """
    Pipeline for SFT on a pretrained model
    """

    # Prepare quantization scheme
    quantization = QuantizationLoader(**QUANTIZATION_CONFIG).load() if USE_QUANTIZATION else None

    # Set PEFT
    peft_configs = []
    if USE_LORA: 
        lora = LoRALoader(**LORA_CONFIG).load()
        peft_configs.append(lora)
    elif USE_PROMPT_TUNNING:
        prompt_tunning = PromptTunningLoader(MODEL, **PROMPT_TUNNING_CONFIG).load()
        peft_configs.append(prompt_tunning)

    # Load Tokenizer
    tokenizer = TokenizerLoader(TOKENIZER if TOKENIZER else MODEL, **TOKENIZER_CONFIG).load()
    # Load Model - Add PEFT
    model = ModelLoader(MODEL, pad_token_id=tokenizer.pad_token_id, quantization=quantization, **MODEL_CONFIG).load()
    # Load PEFT
    model = PEFTLoader(model, peft_configs).load()
    # Load dataset
    dataset = FT_training_dataset(dataset_name=DATASET, tokenizer=tokenizer, **DATASET_CONFIG)

    # Prepare trainer
    trainer = Trainer(
        model,
        dataset,
        tokenizer,
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