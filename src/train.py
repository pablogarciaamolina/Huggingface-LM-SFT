import os

from transformers import TrainingArguments
from peft import PeftConfig
from trl import SFTTrainer
from .data import FT_training_dataset

RESULTS_PATH = "models"

TRAINING_ARGUMENTS = {
    "eval_strategy": "steps",                # Evaluation strategy: evaluate every few steps
    "do_eval": False,                         # Enable evaluation during training
    "gradient_accumulation_steps": 2,        # Accumulate gradients over multiple steps
    "per_device_eval_batch_size": 2,         # Batch size per device during evaluation
    "log_level": "debug",                    # Set logging level to debug for detailed logs
    "logging_steps": 10,                     # Log metrics every 10 steps
    "eval_steps": 25,                        # Evaluate the model every 25 steps
    "save_steps": 25,                        # Save checkpoints every 25 steps
}

class Trainer:
    """
    Main class for training a model, a tokenizer and a dataset. Sets up an environment for training.
    """

    def __init__(
            self,
            model,
            dataset: FT_training_dataset,
            tokenizer,
            peft_config: PeftConfig,
            hyperparameters: dict,
            saving_name: str = "results",
            verbose: bool = True,
            **sft_trainer_specifications
        ) -> None:
        """
        tokenizer_name: If no tokenizer name is provided, it is supposed is the same as the model name.
        """

        # self.model = model
        # self.dataset = dataset
        # self.tokenizer = tokenizer
        self.verbose = verbose

        training_args = TrainingArguments(
            output_dir = os.path.join(RESULTS_PATH, saving_name),
            **{**TRAINING_ARGUMENTS, **hyperparameters}
        )
        self.trainer = SFTTrainer(
            model=model,
            train_dataset=dataset.train_data,
            eval_dataset=dataset.test_data,
            tokenizer=tokenizer,     
            peft_config=peft_config,  
            args=training_args,
            **sft_trainer_specifications
        )

    def train(self) -> None:

        if self.verbose: print("TRAINING ...")
        self.trainer.train()
        if self.verbose: print("DONE")

