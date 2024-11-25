from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
 
LIMA = "GAIR/lima"
ALPACA = "tatsu-lab/alpaca"
ULTRABIN = "ContextualAI/ultrabin_clean_max_chosen_min_rejected_rationalized_instruction_following"
 
class FT_training_dataset():
    """
    Class containing the pre-process of the three datasets we use: tatsu-lab/alpaca, GAIR/lima, ContextualAI/ultrabin_clean_max_chosen_min_rejected_rationalized_instruction_following
    """
 
    def __init__(self, dataset_name: str, tokenizer: PreTrainedTokenizerBase, num_val: int):
        self.dataset_name = dataset_name
        self.num_val = num_val
 
        self.test_data = None
        self.train_data = None
        self.tokenizer = tokenizer
 
        self.format_functions = {
            LIMA: self.format_conversation_lima,
            ALPACA: self.format_conversation_alpaca,
            ULTRABIN: self.format_conversation_ultrabin
        }
 
        self.columns_to_remove = {
            LIMA: ["conversations", "source"],
            ALPACA: ["input", "instruction", "text"],
            ULTRABIN: [
                "source", "prompt", "chosen", "chosen-avg-rating", "chosen-rating",
                "chosen-model", "chosen-rationale", "rejected", "rejected-avg-rating",
                "rejected-rating", "rejected-model", "rejected-rationale", "ranking-attribute"
            ]
        }
 
        self.get_data(self.tokenizer)
 
    def get_data(self, tokenizer):
        dataset = load_dataset(self.dataset_name, split="train")
        dataset = dataset.train_test_split(test_size=0.2)
 
        self.tokenizer = tokenizer
 
        if self.dataset_name not in self.format_functions:
            raise Exception(f"FT_training_dataset doesn't have an implementation for the {self.dataset_name} dataset")
 
        format_fn = self.format_functions[self.dataset_name]
 
        tokenized_dataset = dataset.map(format_fn, batched=True)
 
        tokenized_dataset = self.remove_unwanted_columns(tokenized_dataset)
 
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
 
        test_data = tokenized_dataset['test'].shuffle(seed=42)
        train_data = tokenized_dataset['train'].shuffle(seed=42)
        test_data = test_data.select(range(self.num_val))
 
        self.test_data = test_data
        self.train_data = train_data
 
    def remove_unwanted_columns(self, dataset):
        """
        Remove columns specific to each dataset based on the `columns_to_remove` dictionary.
        """
        if self.dataset_name in self.columns_to_remove:
            columns = self.columns_to_remove[self.dataset_name]
            dataset = dataset.remove_columns(columns)
        return dataset
 
    def format_conversation_lima(self, examples):
        """
        Lima dataset formatting function.
        """
        joined_conversations = [" ".join(conv) if isinstance(conv, list) else conv for conv in examples['conversations']]
        return self.tokenizer(joined_conversations, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
 
    def format_conversation_alpaca(self, examples):
        """
        Alpaca dataset formatting function.
        """
        joined_conversations = ["\nPrompt:" + examples['instruction'][i] + examples['input'][i]+ "\nResponse:" + examples['output'][i] for i in range(len(examples['instruction']))]
        return self.tokenizer(joined_conversations, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
 
    def format_conversation_ultrabin(self, examples):
        """
        Ultrabin dataset formatting function.
        """
        joined_conversations = ["\nPrompt:" + examples['prompt'][i] + "\nResponse:" + examples['chosen'][i][0]["content"] for i in range(len(examples['prompt']))]
        return self.tokenizer(joined_conversations, truncation=True, max_length=512, padding="max_length", return_tensors="pt")