"""Model/tokenizer loading helpers shared across training and eval."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    return tokenizer


def load_base_model():
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )


def load_fresh_peft_model():
    model = load_base_model()
    return get_peft_model(model, LORA_CONFIG)


def load_peft_from_adapter(adapter_path, is_trainable=False):
    base = load_base_model()
    return PeftModel.from_pretrained(base, adapter_path, is_trainable=is_trainable)
