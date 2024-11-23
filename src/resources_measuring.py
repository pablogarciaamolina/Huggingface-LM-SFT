import torch
from transformers import AutoModel, AutoTokenizer
import sys

def estimate_model_memory(model_name: str):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    model_size = sum(p.numel() for p in model.parameters()) * 0.5 / (1024 ** 3) # as for the quantization

    vocab_size = len(tokenizer.get_vocab()) 
    vocab_memory = vocab_size * sys.getsizeof('') 
    vocab_dict_memory = vocab_size * sys.getsizeof('') * 2 
    total_tokenizer_memory = vocab_memory + vocab_dict_memory


    total_memory_estimate = model_size + (total_tokenizer_memory / (1024 ** 3))

    print(f"Modelo cargado: {model_name}")
    print(f"Tamaño estimado del modelo: {model_size:.3f} GB")
    print(f"Memoria total estimada para el tokenizer: {total_tokenizer_memory / (1024 ** 2):.3f} MB")
    print(f"Tamaño estimado total: {total_memory_estimate:.3f} GB")