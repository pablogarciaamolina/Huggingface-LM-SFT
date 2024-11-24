from peft import PromptTuningConfig, LoraConfig, PeftConfig, TaskType
from peft.peft_model import PeftModel
from transformers.modeling_utils import PreTrainedModel


BASE_MODULES = ['k_proj', 'q_proj', 'v_proj', 'o_proj','gate_proj', 'down_proj', 'up_proj']

class PEFTLoader:
    
    def __init__(self, peft_configs: list[PeftConfig], peft_type: str = "composed") -> None:

        self.configs = peft_configs
        self.type = peft_type

    def load(self, model: PreTrainedModel) -> PeftModel:

        for config in self.configs:
            model = PeftModel(model, config)

        return model


class LoRALoader:

    def __init__(
            self,
            lora_alpha: float = 16,             # Scaling factor for LoRA updates
            lora_dropout: float = 0.05,         # Dropout rate applied to LoRA layers
            r: int = 16,                      # Rank of the LoRA decomposition
            bias = "none",               # No bias is added to the LoRA layers
            task_type = TaskType.CAUSAL_LM,     # Specify the task as causal language modeling
            target_modules = BASE_MODULES # Modules to apply LoRA to
        ) -> None:
        """
        lora_alpha: Scaling factor for LoRA updates
        lora_dropout: Dropout rate applied to LoRA layers
        r: Rank of the LoRA decomposition
        bias: Bias added to the LoRA layers
        task_type: Task, as causal language modeling
        target_modules: Modules to apply LoRA to
        """

        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.r = r
        self.bias = bias
        self.task_type = task_type
        self.target_modules = target_modules

    def load(self) -> LoraConfig:

        return LoraConfig(
            lora_alpha = self.lora_alpha,
            lora_dropout = self.lora_dropout,
            r = self.r,
            bias = self.bias,
            task_type = self.task_type,
            target_modules = self.target_modules
        )

class PromptTunningLoader:
    
    def __init__(
            self,
            tokenizer_name_or_path: str | None = None,
            task_type = TaskType.CAUSAL_LM,
            num_virtual_tokens: int = 20
        ) -> None:
        
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.task_type = task_type
        self.num_virtual_tokens = num_virtual_tokens

    def load(self) -> PromptTuningConfig:

        return PromptTuningConfig(
            task_type=self.task_type,
            num_virtual_tokens=self.num_virtual_tokens,
            tokenizer_name_or_path=self.tokenizer_name_or_path
        )