import torch
from transformers import BitsAndBytesConfig


class QuantizationLoader:

    def __init__(
        self,
        load_in_4bit=True,                    # Enable loading the model in 4-bit precision
        bnb_4bit_quant_type="nf4",            # Specify quantization type as Normal Float 4
        bnb_4bit_compute_dtype=getattr(torch, "bfloat16"), # Set computation data type
        bnb_4bit_use_double_quant=True
    ) -> None:
        """
        load_in_4bit: Enable loading the model in 4-bit precision
        bnb_4bit_quant_type: Quantization type
        bnb_4bit_compute_dtype: Computation data type
        bnb_4bit_use_double_quant: Use double quantization for better accuracy
        """

        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

    def load(self) -> BitsAndBytesConfig:
        
        self.config = BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant
        )

        return self.config