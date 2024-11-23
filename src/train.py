import os

from models import ModelLoader, TokenizerLoader
from .data import FT_training_dataset

from transformers import TrainingArguments
from trl import SFTTrainer


TRAINING_ARGUMENTS = {
    "eval_strategy": "steps",                # Evaluation strategy: evaluate every few steps
    "do_eval": False,                         # Enable evaluation during training
    "gradient_accumulation_steps": 2,        # Accumulate gradients over multiple steps
    "per_device_eval_batch_size": 2,         # Batch size per device during evaluation
    "log_level": "debug",                    # Set logging level to debug for detailed logs
    "logging_steps": 10,                     # Log metrics every 10 steps
    "eval_steps": 25,                        # Evaluate the model every 25 steps
    "save_steps": 25,                        # Save checkpoints every 25 steps
    "warmup_steps": 25,                      # Number of warmup steps for learning rate scheduler
}

HYPERPARAMETERS = {
    "learning_rate": 1e-4,                   # Initial learning rate
    "optim": "paged_adamw_8bit",             # Use 8-bit AdamW optimizer for memory efficiency
    "per_device_train_batch_size": 4,        # Batch size per device during training
    "max_steps": 200,                        # Total number of training steps
    "lr_scheduler_type": "linear"            # Use a linear learning rate scheduler
}


class Trainer:
    """
    Main class for training a model, a tokenizer and a dataset. Sets up an environment for training.
    """

    def __init__(
            self,
            model_name,
            dataset_name,
            tokenizer_name: str = None,
            peft_config = None,
            **kargs
        ) -> None:
        """
        tokenizer_name: If no tokenizer name is provided, it is supposed is the same as the model name.
        """

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        self.peft = peft_config

        # Extracting arguments from kargs
        ## tokenizer-related 
        add_eos_token = kargs.get("add_eos_token", True)  # Default to True if not provided
        use_fast = kargs.get("use_fast", True)  # Default to True
        padding_side = kargs.get("padding_side", "left")  # Default to "right"
        ## model-related
        quantization = kargs.get("quantization", None)  # Default to None if not provided
        device_map = kargs.get("device_map", "auto")  # Default to "auto"
        ## dataset-related
        num_val = kargs.get("num_val", 25)  # Default to 1000 validation examples
        ## others
        self.max_seq_len = kargs.get("max_seq_len", 512)

        self.tokenizer = TokenizerLoader(
            self.tokenizer_name,
            add_eos_token=add_eos_token,
            use_fast=use_fast,
            padding_side=padding_side
        ).load()
        self.model = ModelLoader(
            model_name,
            pad_token_id=self.tokenizer.pad_token_id,
            quantization=quantization,
            device_map=device_map
        ).load()
        self.dataset = FT_training_dataset(
            dataset_name=self.dataset_name,
            num_val=num_val
        )

    def train(self) -> None:

        train_model(
            saving_dir=os.path.join("models", self.model_name+"_results"),
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            peft_config=self.peft,
            max_seq_len=self.max_seq_len
        )



def train_model(
        saving_dir: str,
        model,
        tokenizer,
        dataset,
        peft_config,
        max_seq_len,
) -> None:
    """
    Function for trianing the model and saving it following TRAINING_ARGUMENTS parameters
    """

    training_args = TrainingArguments(
        output_dir = saving_dir,
        **{**TRAINING_ARGUMENTS, **HYPERPARAMETERS}
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,                  
        args=training_args,
        max_seq_length=max_seq_len
    )

