import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from transformers import Trainer 
from datasets import load_dataset
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import json
from tqdm import tqdm
from peft import PeftModel
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

class ClassifierTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits[:, -3, [8241, 3782]]        # token_id for "Yes" "No"
        labels = (inputs["labels"][:, -2] == 8241)
        yes_probs = torch.softmax(logits, dim=-1)[:, 0]
        binary_loss = -((labels.float()*torch.log(yes_probs) + (1-labels.float())*torch.log(1-yes_probs))).mean()
        return (binary_loss, outputs) if return_outputs else binary_loss