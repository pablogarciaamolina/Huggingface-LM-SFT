from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel


class ModelType:
    
    QWEN = "Qwen/Qwen2.5-7B"
    RECURRENT_GEMMA = "google/recurrentgemma-9b"
    LLAMA =  "meta-llama/Llama-3.1-8B"
    GEMMA_2B = "google/gemma-2-2b-jpn-it"
    MISTRAL = "mistralai/Mistral-7B-v0.3"
    PHI_2 = "microsoft/phi-2"

class ModelLoader:

    def __init__(
        self,
        name: ModelType,
        pad_token_id,
        quantization: BitsAndBytesConfig | None = None,
        device_map: str = "auto"
    ) -> None:
        """
        name: name or path of the model
        pad_toke_id: Usually `tokenizer.pad_token_id`
        """

        self.name = name
        self.pad_token_id: int = pad_token_id
        self.quantization = quantization
        self.device_map = device_map

    def load(self) -> PreTrainedModel:

       self. model = AutoModelForCausalLM.from_pretrained(
           self.name,
           quantization_config=self.quantization,
           device_map=self.device_map
       )
       self.model.config.pad_token_id = self.pad_token_id

       return self.model