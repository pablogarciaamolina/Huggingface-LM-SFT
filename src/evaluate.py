import os
import json
import subprocess

from tqdm import tqdm
import torch
from datasets import load_dataset

SAVING_EVALUATION_DIR = "evaluations"
SAVING_MEASURE_DIR = "measures"

class IFEVALEvaluator:

    def __init__(self,
            model,
            tokenizer,
            verbose: bool = True,
    ) -> None:

        self.model = model
        self.dataset = self.load_dataset()
        self.tokenizer = tokenizer

        self.verbose = verbose

    def load_dataset(self):

        return load_dataset("google/IFEval")

    def evaluate(
            self,
            save_name: str,
            device: str = "cpu",
            batch_size: int = 8,
            num_batches: int = 5,
            max_length: int = 128
    ) -> None:
        
        
        # Disable gradients to save memory and computation
        self.model.eval()
        torch.set_grad_enabled(False)  # Disable gradient computation globally

        # Prepare the output file
        os.makedirs(SAVING_EVALUATION_DIR, exist_ok=True)
        output_file = os.path.join(SAVING_EVALUATION_DIR, f"{save_name}.json")

        # Batch processing
        batch_size = 8  # Adjust based on your GPU memory capacity
        max_length = 128  # Limit output length to avoid excessive memory usage
        with open(output_file, 'w') as f:
            # Process in batches
            for i in tqdm(range(0, len(self.dataset["train"]['key'][:num_batches*batch_size]), batch_size), desc="Processing Batches", unit="batch"):
                try:
                    if (i + batch_size) > len(self.dataset["train"]):
                        batch_prompts = self.dataset["train"]['prompt'][i:len(self.dataset["train"])]
                    else:
                        batch_prompts = self.dataset["train"]['prompt'][i:i + batch_size]
                    # Tokenize inputs
                    inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                    
                    # Generate responses
                    with torch.no_grad():  # Ensure gradients are disabled during generation
                        if max_length:
                            outputs = self.model.generate(**inputs, max_new_tokens=max_length)
                        else:
                            outputs = self.model.generate(**inputs)
                    # Decode responses
                    responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

                    # Write each response directly to the file
                    for prompt, response in zip(batch_prompts, responses):
                        f.write(json.dumps({"prompt": prompt, "response": response}) + '\n')
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")

        if self.verbose: print(f"Responses saved to {output_file}")

    @staticmethod
    def measure(name: str) -> subprocess.CompletedProcess:

        command = [
            "python3", 
            "-m", "src.instruction_following_eval.evaluation_main",
            "--input_data", os.path.join("src", "instruction_following_eval", "data", "input_data.jsonl"),
            "--input_response_data", os.path.join(SAVING_EVALUATION_DIR, f"{name}.json"),
            "--output_dir", os.path.join(SAVING_MEASURE_DIR, name)
        ]
                
        result = subprocess.run(command, capture_output=True, text=True)

        return result
