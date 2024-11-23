from datasets import load_dataset
from transformers import AutoTokenizer



class FT_training_dataset():
    """
    Class containing the pre-process of the three datasets we use: tatsu-lab/alpaca, GAIR/lima, ContextualAI/ultrabin_clean_max_chosen_min_rejected_rationalized_instruction_following
    """
    def __init__(self, dataset_name: str, model_name: str, num_val: int):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_val = num_val

        self.test_data = None
        self.train_data = None

        self.get_data()

    def get_data(self):

        dataset = load_dataset(self.dataset_name, split="train")
        dataset.train_test_split(test_size=0.2)

        tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        add_eos_token=True,
                        use_fast=True,           
                        padding_side='left'    
                    )
        tokenizer.pad_token = tokenizer.eos_token 

        def format_conversation_lima(examples):
            joined_conversations = [" ".join(conv) if isinstance(conv, list) else conv for conv in examples['conversations']]
            return tokenizer(joined_conversations, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

        def format_conversation_alpaca(examples):
            joined_conversations = [examples['instruction'][i] + examples['input'][i] + examples['output'][i] for i in range(len(examples['instruction']))]
            return tokenizer(joined_conversations, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

        def format_conversation_ultrabin(examples):
            joined_conversations = [examples['prompt'][i] + examples['chosen'][i][0] for i in range(len(examples['prompt']))]
            return tokenizer(joined_conversations, truncation=True, max_length=512, padding="max_length", return_tensors="pt")


        if self.dataset_name == "GAIR/lima":
            tokenized_dataset = dataset.map(format_conversation_lima, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["conversations", "source"])

        elif self.dataset_name == "tatsu-lab/alpaca":
            tokenized_dataset = dataset.map(format_conversation_alpaca, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["input", "instruction", "text"])

        elif self.dataset_name == "ContextualAI/ultrabin_clean_max_chosen_min_rejected_rationalized_instruction_following":
            tokenized_dataset = dataset.map(format_conversation_ultrabin, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["source", "prompt", "chosen", "chosen-avg-rating", "chosen-rating", "chosen-model", "chosen-rationale", "rejected", "rejected-avg-rating", "rejected-rating", "rejected-model", "rejected-rationale", "ranking-attribute"])

        else:
            raise Exception("FT_training_dataset doesn't have an implementation for your dataset")
        
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        test_data = tokenized_dataset['test'].shuffle(seed=42)
        train_data =  tokenized_dataset['train'].shuffle(seed=42)
        test_data = test_data.select(range(self.num_val))

        self.test_data = test_data
        self.train_data = train_data



