from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class TokenizerLoader:

    def __init__(
            self,
            model_name: str,
            add_eos_token: bool = True,
            use_fast: bool = True,
            padding_side: str = 'left'
        ) -> None:
        
        self.model_name = model_name
        self.add_eos_token = add_eos_token
        self.use_fast = use_fast
        self.padding_side = padding_side

    def load(self) -> PreTrainedTokenizerBase:

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            add_eos_token = self.add_eos_token,
            use_fast = self.use_fast,
            padding_side = self.padding_side
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer